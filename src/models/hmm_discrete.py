"""
Hidden Markov Model Discreto para markov-crypto-forecaster.

Implementa HMM con emisiones discretas usando hmmlearn.
Utiliza features cuantizadas para crear un alfabeto finito de observaciones.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple, List
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
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


class DiscreteHMM(MarkovModelBase, ModelValidationMixin):
    """
    Hidden Markov Model con emisiones discretas.
    
    Convierte features continuas a un alfabeto discreto usando cuantización
    y luego entrena un HMM multinomial. Útil cuando se quiere modelar
    comportamientos categóricos del mercado.
    
    Características:
    - Discretización automática de features continuas
    - Múltiples estrategias de cuantización
    - Manejo de alfabetos grandes
    - Interpretación simbólica de estados
    """
    
    def __init__(self, config: Union[Config, dict], n_components: Optional[int] = None):
        """
        Inicializa el modelo HMM Discreto.
        
        Args:
            config: Configuración del sistema
            n_components: Número de estados (sobrescribe configuración)
        """
        super().__init__(config, 'hmm_discrete')
        
        if not HMMLEARN_AVAILABLE:
            raise ImportError("hmmlearn requerido para HMM Discreto")
        
        # Parámetros del modelo
        self.n_components = n_components or self.model_config.get('n_components', 3)
        self.n_iter = self.model_config.get('n_iter', 100)
        self.tol = self.model_config.get('tol', 1e-2)
        
        # Validar parámetros
        self.n_components = self.validate_n_states(self.n_components)
        self.n_states = self.n_components
        
        # Parámetros de discretización
        self.discretization_config = self.config.get('features', {}).get('discretization', {})
        self.n_bins = self.discretization_config.get('n_quantiles', 5)
        self.discretization_strategy = self.discretization_config.get('strategy', 'quantile')
        self.features_to_discretize = self.discretization_config.get('features_to_discretize', [])
        
        # Discretizadores
        self.discretizers = {}
        self.label_encoder = LabelEncoder()
        self.fitted_discretizers = False
        self.n_symbols = None
        
        # Mapeo de símbolos
        self.symbol_mapping = {}
        self.reverse_symbol_mapping = {}
        
        # Métricas de entrenamiento
        self.discretization_info = {}
        self.symbol_frequencies = {}
        
        logger.info(f"HMM Discreto inicializado: {self.n_components} estados, {self.n_bins} bins")
    
    @log_execution_time("fit_hmm_discrete")
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs) -> 'DiscreteHMM':
        """
        Entrena el modelo HMM Discreto.
        
        Args:
            X: Features de entrada [n_samples, n_features]
            y: No utilizado (HMM es no supervisado)
            **kwargs: Argumentos adicionales
            
        Returns:
            Self para method chaining
            
        Raises:
            ValueError: Si los datos no son válidos
        """
        logger.info(f"Entrenando HMM Discreto con {len(X)} muestras")
        
        # Validar entrada
        X_array, _ = self.validate_input(X, y)
        
        # Validaciones adicionales
        n_samples, n_features = X_array.shape
        self.validate_sample_size(n_samples, self.n_components, n_features)
        
        # Guardar nombres de features
        self.feature_names = list(X.columns)
        
        # Determinar features a discretizar
        if not self.features_to_discretize:
            # Si no se especifican, usar todas las numéricas
            self.features_to_discretize = [
                col for col in X.columns 
                if pd.api.types.is_numeric_dtype(X[col])
            ]
            logger.info(f"Discretizando todas las features numéricas: {self.features_to_discretize}")
        
        # Discretizar features
        X_discrete = self._discretize_features(X, fit=True)
        
        # Convertir a secuencia de símbolos
        symbol_sequence = self._features_to_symbols(X_discrete, fit=True)
        
        # Crear y entrenar modelo
        self.model = self._create_hmm_model()
        
        # Entrenar
        try:
            logger.info("Iniciando entrenamiento de HMM Discreto")
            self.model.fit(symbol_sequence.reshape(-1, 1))
            
            # Verificar convergencia
            if hasattr(self.model, 'monitor_'):
                converged = self.model.monitor_.converged
                n_iter_performed = len(self.model.monitor_.history) if self.model.monitor_.history else 0
                
                if converged:
                    logger.info(f"Convergencia alcanzada en {n_iter_performed} iteraciones")
                else:
                    logger.warning(f"No convergió en {n_iter_performed} iteraciones")
            
            # Validar modelo entrenado
            self._validate_trained_model(symbol_sequence)
            
            # Calcular métricas
            self._calculate_training_metrics(symbol_sequence, X_discrete)
            
            self.is_fitted = True
            logger.info("Entrenamiento de HMM Discreto completado exitosamente")
            
        except Exception as e:
            logger.error(f"Error en entrenamiento: {e}")
            raise RuntimeError(f"Entrenamiento falló: {e}")
        
        return self
    
    def _create_hmm_model(self) -> hmm.MultinomialHMM:
        """Crea instancia del modelo HMM."""
        if self.n_symbols is None:
            raise ValueError("Número de símbolos no determinado")
        
        model = hmm.MultinomialHMM(
            n_components=self.n_components,
            n_iter=self.n_iter,
            tol=self.tol,
            verbose=False,
            params='ste',  # start, transition, emission
            init_params='ste',
            random_state=self.random_state
        )
        
        return model
    
    def _discretize_features(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Discretiza features continuas.
        
        Args:
            X: DataFrame con features
            fit: Si ajustar los discretizadores
            
        Returns:
            DataFrame con features discretizadas
        """
        X_discrete = pd.DataFrame(index=X.index)
        
        for feature in self.features_to_discretize:
            if feature not in X.columns:
                logger.warning(f"Feature '{feature}' no encontrada")
                continue
            
            feature_data = X[feature].dropna()
            if len(feature_data) < self.n_bins:
                logger.warning(f"Pocos datos para discretizar '{feature}': {len(feature_data)}")
                continue
            
            if fit:
                # Crear discretizador
                discretizer = KBinsDiscretizer(
                    n_bins=self.n_bins,
                    encode='ordinal',
                    strategy=self.discretization_strategy,
                    subsample=200000  # Para datasets grandes
                )
                
                # Ajustar discretizador
                try:
                    discretized = discretizer.fit_transform(feature_data.values.reshape(-1, 1)).flatten()
                    self.discretizers[feature] = discretizer
                    
                    # Crear serie completa
                    discrete_series = pd.Series(index=X.index, dtype=int)
                    discrete_series.loc[feature_data.index] = discretized.astype(int)
                    X_discrete[f'{feature}_discrete'] = discrete_series
                    
                    # Guardar información de discretización
                    self.discretization_info[feature] = {
                        'n_bins': self.n_bins,
                        'strategy': self.discretization_strategy,
                        'bin_edges': discretizer.bin_edges_[0].tolist(),
                        'n_samples': len(feature_data)
                    }
                    
                    logger.info(f"Feature '{feature}' discretizada en {self.n_bins} bins")
                    
                except Exception as e:
                    logger.error(f"Error discretizando '{feature}': {e}")
                    continue
            
            else:
                # Usar discretizador existente
                if feature in self.discretizers:
                    discretizer = self.discretizers[feature]
                    try:
                        discretized = discretizer.transform(feature_data.values.reshape(-1, 1)).flatten()
                        discrete_series = pd.Series(index=X.index, dtype=int)
                        discrete_series.loc[feature_data.index] = discretized.astype(int)
                        X_discrete[f'{feature}_discrete'] = discrete_series
                    except Exception as e:
                        logger.error(f"Error aplicando discretización a '{feature}': {e}")
                else:
                    logger.warning(f"Discretizador para '{feature}' no encontrado")
        
        if fit:
            self.fitted_discretizers = True
        
        return X_discrete
    
    def _features_to_symbols(self, X_discrete: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """
        Convierte features discretizadas a secuencia de símbolos.
        
        Args:
            X_discrete: DataFrame con features discretizadas
            fit: Si ajustar el codificador de símbolos
            
        Returns:
            Array con secuencia de símbolos
        """
        if X_discrete.empty:
            raise ValueError("No hay features discretizadas")
        
        # Crear tuplas combinando todas las features discretizadas
        # Esto crea un "estado conjunto" de todas las features
        tuples = []
        for idx in X_discrete.index:
            # Obtener valores discretos para esta observación
            values = []
            for col in X_discrete.columns:
                val = X_discrete.loc[idx, col]
                if pd.isna(val):
                    val = -1  # Símbolo especial para NaN
                values.append(int(val))
            tuples.append(tuple(values))
        
        if fit:
            # Ajustar label encoder
            unique_tuples = list(set(tuples))
            self.label_encoder.fit(unique_tuples)
            self.n_symbols = len(self.label_encoder.classes_)
            
            # Crear mapeos para interpretación
            for i, symbol_tuple in enumerate(self.label_encoder.classes_):
                self.symbol_mapping[i] = {
                    'tuple': symbol_tuple,
                    'features': dict(zip(X_discrete.columns, symbol_tuple))
                }
                self.reverse_symbol_mapping[symbol_tuple] = i
            
            logger.info(f"Creados {self.n_symbols} símbolos únicos")
            
            # Calcular frecuencias de símbolos
            symbol_sequence = self.label_encoder.transform(tuples)
            unique_symbols, counts = np.unique(symbol_sequence, return_counts=True)
            self.symbol_frequencies = {
                int(symbol): int(count) for symbol, count in zip(unique_symbols, counts)
            }
            
        else:
            # Transformar usando encoder existente
            try:
                symbol_sequence = self.label_encoder.transform(tuples)
            except ValueError as e:
                # Manejar símbolos no vistos en entrenamiento
                logger.warning(f"Símbolos no vistos en entrenamiento: {e}")
                # Mapear símbolos desconocidos al símbolo más frecuente
                most_frequent_symbol = max(self.symbol_frequencies.items(), key=lambda x: x[1])[0]
                
                symbol_sequence = []
                for tuple_val in tuples:
                    if tuple_val in self.reverse_symbol_mapping:
                        symbol_sequence.append(self.reverse_symbol_mapping[tuple_val])
                    else:
                        symbol_sequence.append(most_frequent_symbol)
                        
                symbol_sequence = np.array(symbol_sequence)
        
        return symbol_sequence
    
    def _validate_trained_model(self, symbol_sequence: np.ndarray):
        """Valida modelo entrenado."""
        try:
            # Verificar log-likelihood
            loglik = self.model.score(symbol_sequence.reshape(-1, 1))
            if np.isnan(loglik) or np.isinf(loglik):
                raise ValueError("Log-likelihood inválido")
            
            # Verificar matriz de transición
            if not np.allclose(self.model.transmat_.sum(axis=1), 1.0, rtol=1e-3):
                raise ValueError("Matriz de transición no normalizada")
            
            # Verificar matriz de emisión
            if not np.allclose(self.model.emissionprob_.sum(axis=1), 1.0, rtol=1e-3):
                raise ValueError("Matriz de emisión no normalizada")
            
            logger.info("Validación del modelo exitosa")
            
        except Exception as e:
            logger.error(f"Modelo no válido: {e}")
            raise
    
    def _calculate_training_metrics(self, symbol_sequence: np.ndarray, X_discrete: pd.DataFrame):
        """Calcula métricas de entrenamiento."""
        try:
            # Log-likelihood
            loglik = self.model.score(symbol_sequence.reshape(-1, 1))
            
            # Criterios de información
            info_criteria = self.calculate_information_criteria(
                pd.DataFrame(symbol_sequence, columns=['symbols'])
            )
            
            # Estadísticas de estados
            states = self.model.predict(symbol_sequence.reshape(-1, 1))
            state_counts = np.bincount(states, minlength=self.n_components)
            state_frequencies = state_counts / len(states)
            
            # Estadísticas de símbolos
            symbol_diversity = len(np.unique(symbol_sequence)) / self.n_symbols
            
            self.training_metrics = {
                'log_likelihood': loglik,
                'perplexity': np.exp(-loglik / len(symbol_sequence)),
                'state_frequencies': state_frequencies.tolist(),
                'n_unique_symbols': len(np.unique(symbol_sequence)),
                'symbol_diversity': symbol_diversity,
                'discretization_info': self.discretization_info,
                **info_criteria
            }
            
            self.model_info = {
                'n_components': self.n_components,
                'n_symbols': self.n_symbols,
                'n_features_discretized': len(self.features_to_discretize),
                'n_training_samples': len(symbol_sequence),
                'transition_matrix': self.model.transmat_.tolist(),
                'emission_matrix': self.model.emissionprob_.tolist(),
                'start_probabilities': self.model.startprob_.tolist()
            }
            
            logger.info(f"Métricas calculadas: LogLik={loglik:.2f}, Diversidad símbolos={symbol_diversity:.3f}")
            
        except Exception as e:
            logger.error(f"Error calculando métricas: {e}")
            self.training_metrics = {'error': str(e)}
    
    def predict_states(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predice secuencia de estados más probable.
        
        Args:
            X: Features de entrada
            
        Returns:
            Array con secuencia de estados
        """
        if not self.is_fitted:
            raise ValueError("Modelo debe estar entrenado")
        
        X_discrete = self._discretize_features(X, fit=False)
        symbol_sequence = self._features_to_symbols(X_discrete, fit=False)
        
        states = self.model.predict(symbol_sequence.reshape(-1, 1))
        return states
    
    def predict_state_probabilities(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calcula probabilidades de estado.
        
        Args:
            X: Features de entrada
            
        Returns:
            Array con probabilidades [n_samples, n_states]
        """
        if not self.is_fitted:
            raise ValueError("Modelo debe estar entrenado")
        
        X_discrete = self._discretize_features(X, fit=False)
        symbol_sequence = self._features_to_symbols(X_discrete, fit=False)
        
        probs = self.model.predict_proba(symbol_sequence.reshape(-1, 1))
        return probs
    
    def sample(self, n_samples: int, random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genera muestras del modelo.
        
        Args:
            n_samples: Número de muestras
            random_state: Semilla aleatoria
            
        Returns:
            Tupla (símbolos, estados)
        """
        if not self.is_fitted:
            raise ValueError("Modelo debe estar entrenado")
        
        if random_state is not None:
            np.random.seed(random_state)
        
        symbols, states = self.model.sample(n_samples)
        return symbols.flatten(), states
    
    def score(self, X: pd.DataFrame) -> float:
        """
        Calcula log-likelihood.
        
        Args:
            X: Features de entrada
            
        Returns:
            Log-likelihood
        """
        if not self.is_fitted:
            raise ValueError("Modelo debe estar entrenado")
        
        X_discrete = self._discretize_features(X, fit=False)
        symbol_sequence = self._features_to_symbols(X_discrete, fit=False)
        
        return self.model.score(symbol_sequence.reshape(-1, 1))
    
    def get_n_parameters(self) -> int:
        """
        Calcula número de parámetros.
        
        Returns:
            Número de parámetros
        """
        if not self.is_fitted:
            return 0
        
        n_states = self.n_components
        n_symbols = self.n_symbols
        
        # Transiciones: n_states * (n_states - 1)
        transition_params = n_states * (n_states - 1)
        
        # Probabilidades iniciales: n_states - 1
        start_params = n_states - 1
        
        # Emisiones: n_states * (n_symbols - 1)
        emission_params = n_states * (n_symbols - 1)
        
        return transition_params + start_params + emission_params
    
    def get_transition_matrix(self) -> Optional[np.ndarray]:
        """Obtiene matriz de transición."""
        if not self.is_fitted:
            return None
        return self.model.transmat_
    
    def get_emission_parameters(self) -> Optional[Dict[str, Any]]:
        """Obtiene parámetros de emisión."""
        if not self.is_fitted:
            return None
        
        return {
            'emission_matrix': self.model.emissionprob_.tolist(),
            'n_symbols': self.n_symbols,
            'symbol_mapping': self.symbol_mapping
        }
    
    def interpret_states(self) -> Dict[int, Dict[str, Any]]:
        """
        Interpreta estados del modelo.
        
        Returns:
            Dict con interpretación de cada estado
        """
        if not self.is_fitted:
            raise ValueError("Modelo debe estar entrenado")
        
        state_interpretation = {}
        emission_probs = self.model.emissionprob_
        
        for state in range(self.n_components):
            # Encontrar símbolos más probables para este estado
            state_emissions = emission_probs[state]
            top_symbol_indices = np.argsort(state_emissions)[::-1][:5]
            
            top_symbols = []
            for symbol_idx in top_symbol_indices:
                prob = state_emissions[symbol_idx]
                symbol_info = self.symbol_mapping.get(symbol_idx, {})
                top_symbols.append({
                    'symbol_id': int(symbol_idx),
                    'probability': float(prob),
                    'features': symbol_info.get('features', {}),
                    'tuple': symbol_info.get('tuple', ())
                })
            
            # Estadísticas del estado
            state_prob = np.mean(emission_probs[state])
            entropy = -np.sum(state_emissions * np.log(state_emissions + 1e-10))
            
            state_interpretation[state] = {
                'top_symbols': top_symbols,
                'average_emission_prob': float(state_prob),
                'entropy': float(entropy),
                'most_likely_symbol': int(top_symbol_indices[0]),
                'most_likely_features': self.symbol_mapping.get(int(top_symbol_indices[0]), {}).get('features', {})
            }
        
        return state_interpretation
    
    def decode_symbols_to_features(self, symbols: np.ndarray) -> pd.DataFrame:
        """
        Decodifica símbolos de vuelta a features discretizadas.
        
        Args:
            symbols: Array de símbolos
            
        Returns:
            DataFrame con features decodificadas
        """
        if not self.is_fitted:
            raise ValueError("Modelo debe estar entrenado")
        
        decoded_features = []
        feature_names = [f"{feat}_discrete" for feat in self.features_to_discretize]
        
        for symbol in symbols:
            if symbol in self.symbol_mapping:
                symbol_info = self.symbol_mapping[symbol]
                decoded_features.append(list(symbol_info['tuple']))
            else:
                # Símbolo desconocido
                decoded_features.append([np.nan] * len(feature_names))
        
        return pd.DataFrame(decoded_features, columns=feature_names)


if __name__ == "__main__":
    # Ejemplo de uso
    try:
        from ..config_loader import load_config
        
        config = load_config("config/binance_spot_example.yaml")
        
        # Crear datos de ejemplo con patrones discretos claros
        np.random.seed(42)
        n_samples = 400
        
        # Simular diferentes regímenes de mercado
        regime_changes = [0, 100, 250, 400]
        data = []
        states_true = []
        
        for i, (start, end) in enumerate(zip(regime_changes[:-1], regime_changes[1:])):
            regime_size = end - start
            
            if i == 0:  # Régimen bajista: retornos negativos, alta vol
                returns = np.random.normal(-0.01, 0.02, regime_size)
                volatility = np.random.normal(0.03, 0.01, regime_size)
                volume = np.random.normal(0.8, 0.2, regime_size)
            elif i == 1:  # Régimen alcista: retornos positivos, vol media
                returns = np.random.normal(0.015, 0.015, regime_size)
                volatility = np.random.normal(0.02, 0.005, regime_size)
                volume = np.random.normal(1.2, 0.3, regime_size)
            else:  # Régimen lateral: retornos cerca de cero, baja vol
                returns = np.random.normal(0.001, 0.01, regime_size)
                volatility = np.random.normal(0.015, 0.003, regime_size)
                volume = np.random.normal(1.0, 0.15, regime_size)
            
            regime_data = np.column_stack([returns, volatility, volume])
            data.append(regime_data)
            states_true.extend([i] * regime_size)
        
        X = pd.DataFrame(
            np.vstack(data),
            columns=['log_return_1', 'volatility', 'volume_zscore']
        )
        
        print(f"Datos de ejemplo creados: {len(X)} muestras")
        print(X.head())
        
        # Configurar features a discretizar
        config_discrete = config.copy()
        config_discrete['features']['discretization']['features_to_discretize'] = ['log_return_1', 'volatility', 'volume_zscore']
        config_discrete['features']['discretization']['n_quantiles'] = 4
        
        # Entrenar modelo
        model = DiscreteHMM(config_discrete, n_components=3)
        model.fit(X)
        
        print(f"\nModelo entrenado exitosamente")
        print(f"Número de símbolos únicos: {model.n_symbols}")
        
        # Predecir estados
        predicted_states = model.predict_states(X)
        
        # Comparar con estados verdaderos
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(states_true, predicted_states)
        print(f"Adjusted Rand Index: {ari:.3f}")
        
        # Métricas del modelo
        print(f"\nMétricas principales:")
        metrics = model.training_metrics
        print(f"  Log-likelihood: {metrics['log_likelihood']:.2f}")
        print(f"  Diversidad símbolos: {metrics['symbol_diversity']:.3f}")
        print(f"  AIC: {metrics['aic']:.2f}")
        print(f"  BIC: {metrics['bic']:.2f}")
        
        # Interpretación de estados
        interpretation = model.interpret_states()
        print(f"\nInterpretación de estados:")
        for state, info in interpretation.items():
            most_likely = info['most_likely_features']
            print(f"  Estado {state}: {most_likely}")
        
        # Información de discretización
        print(f"\nInformación de discretización:")
        for feature, info in model.discretization_info.items():
            print(f"  {feature}: {info['n_bins']} bins, estrategia '{info['strategy']}'")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
