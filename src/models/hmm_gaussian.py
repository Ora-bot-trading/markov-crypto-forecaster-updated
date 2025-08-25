"""
Hidden Markov Model Gaussiano para markov-crypto-forecaster.

Implementa HMM con emisiones Gaussianas multivariantes usando hmmlearn.
Optimizado para series temporales financieras con múltiples features.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple, List
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

try:
    from hmmlearn import hmm
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False
    warnings.warn("hmmlearn no disponible. Instalar con: pip install hmmlearn")

from .base import MarkovModelBase, ModelValidationMixin
from ..config_loader import Config
from ..logging_utils import get_logger, log_execution_time

logger = get_logger(__name__)


class GaussianHMM(MarkovModelBase, ModelValidationMixin):
    """
    Hidden Markov Model con emisiones Gaussianas multivariantes.
    
    Utiliza hmmlearn para implementar HMM Gaussiano con algoritmo EM
    (Baum-Welch) para entrenamiento y algoritmo de Viterbi para decodificación.
    
    Características:
    - Emisiones Gaussianas multivariantes
    - Múltiples tipos de covarianza (full, diag, spherical, tied)
    - Inicialización inteligente con K-means
    - Validación de convergencia
    - Regularización automática
    """
    
    def __init__(self, config: Union[Config, dict], n_components: Optional[int] = None):
        """
        Inicializa el modelo HMM Gaussiano.
        
        Args:
            config: Configuración del sistema
            n_components: Número de estados (sobrescribe configuración)
        """
        super().__init__(config, 'hmm_gaussian')
        
        if not HMMLEARN_AVAILABLE:
            raise ImportError("hmmlearn requerido para HMM Gaussiano")
        
        # Parámetros del modelo
        self.n_components = n_components or self.model_config.get('n_components', 3)
        self.covariance_type = self.model_config.get('covariance_type', 'full')
        self.n_iter = self.model_config.get('n_iter', 100)
        self.tol = self.model_config.get('tol', 1e-2)
        self.init_params = self.model_config.get('init_params', 'kmeans')
        
        # Validar parámetros
        self.n_components = self.validate_n_states(self.n_components)
        self.n_states = self.n_components
        
        # Configuración de regularización
        self.min_covar = self.model_config.get('min_covar', 1e-3)
        self.regularization = self.model_config.get('regularization', True)
        
        # Scaler para normalización
        self.scaler = StandardScaler()
        self.fitted_scaler = False
        
        # Métricas de entrenamiento
        self.convergence_history = []
        self.initialization_attempts = 0
        
        logger.info(f"HMM Gaussiano inicializado: {self.n_components} estados, covariance={self.covariance_type}")
    
    @log_execution_time("fit_hmm_gaussian")
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
           normalize: bool = True, **kwargs) -> 'GaussianHMM':
        """
        Entrena el modelo HMM Gaussiano.
        
        Args:
            X: Features de entrada [n_samples, n_features]
            y: No utilizado (HMM es no supervisado)
            normalize: Si normalizar features
            **kwargs: Argumentos adicionales
            
        Returns:
            Self para method chaining
            
        Raises:
            ValueError: Si los datos no son válidos
            RuntimeError: Si el entrenamiento falla
        """
        logger.info(f"Entrenando HMM Gaussiano con {len(X)} muestras")
        
        # Validar entrada
        X_array, _ = self.validate_input(X, y)
        
        # Validaciones adicionales
        n_samples, n_features = X_array.shape
        self.validate_sample_size(n_samples, self.n_components, n_features)
        self.validate_features_for_states(n_features, self.n_components)
        
        # Guardar nombres de features
        self.feature_names = list(X.columns)
        
        # Preprocesamiento
        if normalize:
            X_scaled = self._preprocess_data(X_array, fit_scaler=True)
        else:
            X_scaled = X_array
            logger.info("Entrenamiento sin normalización")
        
        # Crear y configurar modelo
        self.model = self._create_hmm_model()
        
        # Entrenar con múltiples intentos si es necesario
        success = False
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                self.initialization_attempts += 1
                logger.info(f"Intento de entrenamiento {attempt + 1}/{max_attempts}")
                
                # Inicialización personalizada
                if self.init_params == 'kmeans':
                    self._initialize_with_kmeans(X_scaled)
                
                # Entrenar modelo
                self.model.fit(X_scaled)
                
                # Verificar convergencia
                if hasattr(self.model, 'monitor_'):
                    converged = self.model.monitor_.converged
                    n_iter_performed = len(self.model.monitor_.history)
                    final_loglik = self.model.monitor_.history[-1] if self.model.monitor_.history else np.nan
                    
                    self.convergence_history.append({
                        'attempt': attempt + 1,
                        'converged': converged,
                        'n_iter': n_iter_performed,
                        'final_loglik': final_loglik,
                        'history': self.model.monitor_.history.copy()
                    })
                    
                    if converged:
                        logger.info(f"Convergencia alcanzada en {n_iter_performed} iteraciones")
                        success = True
                        break
                    else:
                        logger.warning(f"No convergió en {n_iter_performed} iteraciones")
                        if attempt < max_attempts - 1:
                            logger.info("Reintentando con diferentes parámetros...")
                            self._adjust_parameters_for_retry()
                        else:
                            logger.warning("Usando modelo sin convergencia completa")
                            success = True  # Aceptar modelo sin convergencia en último intento
                            break
                else:
                    # Si no hay monitor, asumir éxito
                    logger.info("Entrenamiento completado (sin monitor de convergencia)")
                    success = True
                    break
                    
            except Exception as e:
                logger.error(f"Error en intento {attempt + 1}: {e}")
                if attempt == max_attempts - 1:
                    raise RuntimeError(f"Entrenamiento falló después de {max_attempts} intentos: {e}")
                continue
        
        if not success:
            raise RuntimeError("No se pudo entrenar el modelo exitosamente")
        
        # Validar modelo entrenado
        self._validate_trained_model(X_scaled)
        
        # Calcular métricas de entrenamiento
        self._calculate_training_metrics(X_scaled)
        
        self.is_fitted = True
        logger.info("Entrenamiento de HMM Gaussiano completado exitosamente")
        
        return self
    
    def _create_hmm_model(self) -> hmm.GaussianHMM:
        """Crea instancia del modelo HMM."""
        model = hmm.GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            tol=self.tol,
            verbose=False,
            params='stmc',  # start, transition, means, covariance
            init_params='',  # No inicializar automáticamente
            random_state=self.random_state
        )
        
        # Configurar regularización
        if hasattr(model, 'min_covar'):
            model.min_covar = self.min_covar
        
        return model
    
    def _preprocess_data(self, X: np.ndarray, fit_scaler: bool = False) -> np.ndarray:
        """
        Preprocesa datos (normalización, etc.).
        
        Args:
            X: Datos de entrada
            fit_scaler: Si ajustar el scaler
            
        Returns:
            Datos preprocesados
        """
        # Manejar NaN
        if np.isnan(X).any():
            logger.warning("Datos contienen NaN, rellenando con mediana")
            X = pd.DataFrame(X).fillna(method='ffill').fillna(method='bfill').values
        
        # Normalización
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
            self.fitted_scaler = True
            logger.info("Scaler ajustado para normalización")
        elif self.fitted_scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
            logger.warning("No se aplicó normalización (scaler no ajustado)")
        
        return X_scaled
    
    def _initialize_with_kmeans(self, X: np.ndarray):
        """
        Inicializa parámetros del HMM usando K-means.
        
        Args:
            X: Datos de entrenamiento preprocesados
        """
        logger.info("Inicializando HMM con K-means")
        
        try:
            # K-means clustering
            kmeans = KMeans(
                n_clusters=self.n_components,
                random_state=self.random_state,
                n_init=10
            )
            cluster_labels = kmeans.fit_predict(X)
            
            # Inicializar medias con centroides de K-means
            self.model.means_ = kmeans.cluster_centers_
            
            # Inicializar covarianzas
            covariances = []
            for i in range(self.n_components):
                cluster_data = X[cluster_labels == i]
                if len(cluster_data) > 1:
                    if self.covariance_type == 'full':
                        cov = np.cov(cluster_data.T) + np.eye(X.shape[1]) * self.min_covar
                    elif self.covariance_type == 'diag':
                        cov = np.var(cluster_data, axis=0) + self.min_covar
                    elif self.covariance_type == 'spherical':
                        cov = np.mean(np.var(cluster_data, axis=0)) + self.min_covar
                    else:  # tied
                        cov = np.cov(X.T) + np.eye(X.shape[1]) * self.min_covar
                else:
                    # Cluster vacío o con una sola muestra
                    if self.covariance_type == 'full':
                        cov = np.eye(X.shape[1]) * self.min_covar
                    elif self.covariance_type == 'diag':
                        cov = np.ones(X.shape[1]) * self.min_covar
                    elif self.covariance_type == 'spherical':
                        cov = self.min_covar
                    else:  # tied
                        cov = np.eye(X.shape[1]) * self.min_covar
                
                covariances.append(cov)
            
            if self.covariance_type == 'tied':
                self.model.covars_ = covariances[0]  # Una sola matriz para todos
            else:
                self.model.covars_ = np.array(covariances)
            
            # Inicializar probabilidades de estado inicial
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            startprob = np.zeros(self.n_components)
            for label, count in zip(unique_labels, counts):
                startprob[label] = count / len(cluster_labels)
            
            # Regularizar para evitar probabilidades cero
            startprob += 1e-6
            startprob /= startprob.sum()
            self.model.startprob_ = startprob
            
            # Inicializar matriz de transición (uniforme + ruido)
            transmat = np.full((self.n_components, self.n_components), 1.0 / self.n_components)
            transmat += np.random.normal(0, 0.01, transmat.shape)
            transmat = np.abs(transmat)  # Asegurar valores positivos
            transmat = transmat / transmat.sum(axis=1, keepdims=True)  # Normalizar filas
            self.model.transmat_ = transmat
            
            logger.info("Inicialización con K-means completada")
            
        except Exception as e:
            logger.warning(f"Error en inicialización K-means: {e}. Usando inicialización por defecto")
            # Fallback a inicialización simple
            self._simple_initialization(X)
    
    def _simple_initialization(self, X: np.ndarray):
        """Inicialización simple de respaldo."""
        n_features = X.shape[1]
        
        # Medias aleatorias cerca de la media global
        global_mean = np.mean(X, axis=0)
        global_std = np.std(X, axis=0)
        self.model.means_ = np.random.normal(
            global_mean, global_std * 0.5, (self.n_components, n_features)
        )
        
        # Covarianzas iniciales
        if self.covariance_type == 'full':
            self.model.covars_ = np.array([
                np.eye(n_features) * self.min_covar for _ in range(self.n_components)
            ])
        elif self.covariance_type == 'diag':
            self.model.covars_ = np.full((self.n_components, n_features), self.min_covar)
        elif self.covariance_type == 'spherical':
            self.model.covars_ = np.full(self.n_components, self.min_covar)
        else:  # tied
            self.model.covars_ = np.eye(n_features) * self.min_covar
        
        # Probabilidades uniformes
        self.model.startprob_ = np.full(self.n_components, 1.0 / self.n_components)
        self.model.transmat_ = np.full(
            (self.n_components, self.n_components), 1.0 / self.n_components
        )
    
    def _adjust_parameters_for_retry(self):
        """Ajusta parámetros para reintento de entrenamiento."""
        # Incrementar tolerancia
        self.model.tol *= 10
        # Incrementar regularización
        self.model.min_covar *= 2
        logger.info(f"Parámetros ajustados: tol={self.model.tol}, min_covar={self.model.min_covar}")
    
    def _validate_trained_model(self, X: np.ndarray):
        """Valida que el modelo entrenado es correcto."""
        try:
            # Verificar que se pueden calcular log-likelihoods
            loglik = self.model.score(X)
            if np.isnan(loglik) or np.isinf(loglik):
                raise ValueError("Log-likelihood es NaN o infinito")
            
            # Verificar matriz de transición
            if not np.allclose(self.model.transmat_.sum(axis=1), 1.0, rtol=1e-3):
                raise ValueError("Matriz de transición no normalizada")
            
            # Verificar probabilidades iniciales
            if not np.allclose(self.model.startprob_.sum(), 1.0, rtol=1e-3):
                raise ValueError("Probabilidades iniciales no normalizadas")
            
            logger.info("Validación del modelo entrenado exitosa")
            
        except Exception as e:
            logger.error(f"Modelo entrenado no válido: {e}")
            raise
    
    def _calculate_training_metrics(self, X: np.ndarray):
        """Calcula métricas de entrenamiento."""
        try:
            # Log-likelihood
            loglik = self.model.score(X)
            
            # Criterios de información
            info_criteria = self.calculate_information_criteria(
                pd.DataFrame(X, columns=self.feature_names)
            )
            
            # Estabilidad de estados
            states = self.model.predict(X)
            state_counts = np.bincount(states, minlength=self.n_components)
            state_frequencies = state_counts / len(states)
            min_state_freq = np.min(state_frequencies)
            
            self.training_metrics = {
                'log_likelihood': loglik,
                'perplexity': np.exp(-loglik / len(X)),
                'state_frequencies': state_frequencies.tolist(),
                'min_state_frequency': min_state_freq,
                'convergence_history': self.convergence_history,
                'initialization_attempts': self.initialization_attempts,
                **info_criteria
            }
            
            # Información del modelo
            self.model_info = {
                'n_components': self.n_components,
                'covariance_type': self.covariance_type,
                'n_features': len(self.feature_names),
                'n_training_samples': len(X),
                'transition_matrix': self.model.transmat_.tolist(),
                'start_probabilities': self.model.startprob_.tolist()
            }
            
            logger.info(f"Métricas calculadas: LogLik={loglik:.2f}, Min freq estado={min_state_freq:.3f}")
            
        except Exception as e:
            logger.error(f"Error calculando métricas: {e}")
            self.training_metrics = {'error': str(e)}
    
    def predict_states(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predice secuencia de estados más probable (Viterbi).
        
        Args:
            X: Features de entrada
            
        Returns:
            Array con secuencia de estados [n_samples]
        """
        if not self.is_fitted:
            raise ValueError("Modelo debe estar entrenado")
        
        X_array, _ = self.validate_input(X)
        X_scaled = self._preprocess_data(X_array, fit_scaler=False)
        
        states = self.model.predict(X_scaled)
        return states
    
    def predict_state_probabilities(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calcula probabilidades de estado para cada observación.
        
        Args:
            X: Features de entrada
            
        Returns:
            Array con probabilidades [n_samples, n_states]
        """
        if not self.is_fitted:
            raise ValueError("Modelo debe estar entrenado")
        
        X_array, _ = self.validate_input(X)
        X_scaled = self._preprocess_data(X_array, fit_scaler=False)
        
        # Usar algoritmo forward-backward para obtener probabilidades suavizadas
        log_probs = self.model.predict_proba(X_scaled)
        return log_probs
    
    def sample(self, n_samples: int, random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genera muestras del modelo.
        
        Args:
            n_samples: Número de muestras a generar
            random_state: Semilla aleatoria
            
        Returns:
            Tupla (observaciones, estados)
        """
        if not self.is_fitted:
            raise ValueError("Modelo debe estar entrenado")
        
        if random_state is not None:
            np.random.seed(random_state)
        
        X_sampled, states = self.model.sample(n_samples)
        
        # Desnormalizar si es necesario
        if self.fitted_scaler:
            X_sampled = self.scaler.inverse_transform(X_sampled)
        
        return X_sampled, states
    
    def score(self, X: pd.DataFrame) -> float:
        """
        Calcula log-likelihood de los datos.
        
        Args:
            X: Features de entrada
            
        Returns:
            Log-likelihood total
        """
        if not self.is_fitted:
            raise ValueError("Modelo debe estar entrenado")
        
        X_array, _ = self.validate_input(X)
        X_scaled = self._preprocess_data(X_array, fit_scaler=False)
        
        return self.model.score(X_scaled)
    
    def get_n_parameters(self) -> int:
        """
        Calcula número de parámetros del modelo.
        
        Returns:
            Número de parámetros
        """
        if not self.is_fitted:
            return 0
        
        n_features = len(self.feature_names)
        n_states = self.n_components
        
        # Parámetros de transición: n_states * (n_states - 1)
        transition_params = n_states * (n_states - 1)
        
        # Probabilidades iniciales: n_states - 1
        start_params = n_states - 1
        
        # Parámetros de emisión (medias): n_states * n_features
        means_params = n_states * n_features
        
        # Parámetros de covarianza
        if self.covariance_type == 'full':
            # Matriz completa: n_features * (n_features + 1) / 2 por estado
            covar_params = n_states * n_features * (n_features + 1) // 2
        elif self.covariance_type == 'diag':
            # Diagonal: n_features por estado
            covar_params = n_states * n_features
        elif self.covariance_type == 'spherical':
            # Un parámetro por estado
            covar_params = n_states
        else:  # tied
            # Una matriz compartida
            covar_params = n_features * (n_features + 1) // 2
        
        total_params = transition_params + start_params + means_params + covar_params
        
        return total_params
    
    def get_transition_matrix(self) -> Optional[np.ndarray]:
        """
        Obtiene matriz de transición.
        
        Returns:
            Matriz de transición [n_states, n_states]
        """
        if not self.is_fitted:
            return None
        
        return self.model.transmat_
    
    def get_emission_parameters(self) -> Optional[Dict[str, Any]]:
        """
        Obtiene parámetros de emisión.
        
        Returns:
            Dict con medias y covarianzas
        """
        if not self.is_fitted:
            return None
        
        return {
            'means': self.model.means_.tolist(),
            'covariances': self.model.covars_.tolist(),
            'covariance_type': self.covariance_type
        }
    
    def decode_most_likely_sequence(self, X: pd.DataFrame) -> Tuple[float, np.ndarray]:
        """
        Decodifica la secuencia de estados más probable (Viterbi completo).
        
        Args:
            X: Features de entrada
            
        Returns:
            Tupla (log_likelihood, secuencia_estados)
        """
        if not self.is_fitted:
            raise ValueError("Modelo debe estar entrenado")
        
        X_array, _ = self.validate_input(X)
        X_scaled = self._preprocess_data(X_array, fit_scaler=False)
        
        loglik, states = self.model.decode(X_scaled)
        return loglik, states
    
    def get_state_means(self) -> np.ndarray:
        """
        Obtiene medias de cada estado.
        
        Returns:
            Array con medias [n_states, n_features]
        """
        if not self.is_fitted:
            raise ValueError("Modelo debe estar entrenado")
        
        return self.model.means_
    
    def interpret_states(self) -> Dict[int, Dict[str, Any]]:
        """
        Interpreta los estados del modelo.
        
        Returns:
            Dict con interpretación de cada estado
        """
        if not self.is_fitted:
            raise ValueError("Modelo debe estar entrenado")
        
        state_interpretation = {}
        means = self.get_state_means()
        
        for state in range(self.n_components):
            state_mean = means[state]
            
            # Encontrar features más características
            if self.fitted_scaler:
                # Desnormalizar para interpretación
                state_mean_original = self.scaler.inverse_transform(state_mean.reshape(1, -1))[0]
            else:
                state_mean_original = state_mean
            
            feature_ranking = np.argsort(np.abs(state_mean))[::-1]
            top_features = [self.feature_names[i] for i in feature_ranking[:3]]
            
            state_interpretation[state] = {
                'mean_values': state_mean_original.tolist(),
                'top_features': top_features,
                'feature_values': {
                    self.feature_names[i]: state_mean_original[i] 
                    for i in feature_ranking[:5]
                }
            }
        
        return state_interpretation


if __name__ == "__main__":
    # Ejemplo de uso
    try:
        from ..config_loader import load_config
        
        config = load_config("config/binance_spot_example.yaml")
        
        # Crear datos de ejemplo
        np.random.seed(42)
        n_samples = 500
        n_features = 4
        
        # Simular régimes de mercado
        regime_changes = [0, 150, 300, 450, n_samples]
        states_true = []
        data = []
        
        for i, (start, end) in enumerate(zip(regime_changes[:-1], regime_changes[1:])):
            regime_size = end - start
            
            if i == 0:  # Régimen bajista
                mean = [-0.001, 0.02, -0.5, 0.8]
                cov = np.eye(n_features) * 0.01
            elif i == 1:  # Régimen alcista
                mean = [0.002, 0.01, 0.5, 1.2]
                cov = np.eye(n_features) * 0.005
            else:  # Régimen lateral
                mean = [0.0, 0.015, 0.0, 1.0]
                cov = np.eye(n_features) * 0.008
            
            regime_data = np.random.multivariate_normal(mean, cov, regime_size)
            data.append(regime_data)
            states_true.extend([i] * regime_size)
        
        X = pd.DataFrame(
            np.vstack(data),
            columns=['log_return_1', 'volatility', 'rsi_normalized', 'volume_zscore']
        )
        
        print(f"Datos de ejemplo creados: {len(X)} muestras, {len(X.columns)} features")
        
        # Entrenar modelo
        model = GaussianHMM(config, n_components=3)
        model.fit(X)
        
        print(f"\nModelo entrenado exitosamente")
        print(f"Estados predichos vs reales:")
        
        # Predecir estados
        predicted_states = model.predict_states(X)
        
        # Comparar con estados verdaderos (solo para validación)
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(states_true, predicted_states)
        print(f"Adjusted Rand Index: {ari:.3f}")
        
        # Métricas del modelo
        print(f"\nMétricas del modelo:")
        for metric, value in model.training_metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
        
        # Interpretación de estados
        interpretation = model.interpret_states()
        print(f"\nInterpretación de estados:")
        for state, info in interpretation.items():
            print(f"  Estado {state}: features principales = {info['top_features']}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
