"""
Markov Switching Autoregressive Models para markov-crypto-forecaster.

Implementa modelos AR con cambio de régimen usando statsmodels.
Permite modelar series temporales con diferentes comportamientos
autorregresivos en diferentes estados del mercado.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple, List
import warnings

try:
    from statsmodels.tsa.regime_switching import MarkovAutoregression
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels no disponible. Instalar con: pip install statsmodels")

from .base import MarkovModelBase, ModelValidationMixin
from ..config_loader import Config
from ..logging_utils import get_logger, log_execution_time

logger = get_logger(__name__)


class MarkovSwitchingAR(MarkovModelBase, ModelValidationMixin):
    """
    Markov Switching Autoregressive Model.
    
    Implementa modelos AR con cambio de régimen donde los parámetros
    autorregresivos pueden cambiar según el estado oculto del mercado.
    
    Características:
    - Múltiples órdenes AR por régimen
    - Varianza switching opcional
    - Detección automática de estacionariedad
    - Diagnósticos de residuos
    - Predicción probabilística
    """
    
    def __init__(self, config: Union[Config, dict], 
                 k_regimes: Optional[int] = None,
                 order: Optional[int] = None):
        """
        Inicializa el modelo MS-AR.
        
        Args:
            config: Configuración del sistema
            k_regimes: Número de regímenes (sobrescribe configuración)
            order: Orden AR (sobrescribe configuración)
        """
        super().__init__(config, 'ms_ar')
        
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels requerido para Markov Switching AR")
        
        # Parámetros del modelo
        self.k_regimes = k_regimes or self.model_config.get('k_regimes', 2)
        self.order = order or self.model_config.get('order', 1)
        self.switching_ar = self.model_config.get('switching_ar', True)
        self.switching_variance = self.model_config.get('switching_variance', True)
        
        # Validar parámetros
        self.k_regimes = self.validate_n_states(self.k_regimes)
        self.n_states = self.k_regimes
        
        if not isinstance(self.order, int) or self.order < 1:
            raise ValueError("order debe ser un entero positivo")
        
        # Configuración de entrenamiento
        self.method = self.model_config.get('method', 'mle')
        self.maxiter = self.model_config.get('maxiter', 500)
        self.tolerance = self.model_config.get('tolerance', 1e-4)
        
        # Datos de entrenamiento
        self.y_training = None
        self.exog_training = None
        
        # Diagnósticos
        self.stationarity_tests = {}
        self.residual_diagnostics = {}
        
        logger.info(f"MS-AR inicializado: {self.k_regimes} regímenes, orden AR={self.order}")
    
    @log_execution_time("fit_ms_ar")
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
           target_column: str = 'log_return_1', **kwargs) -> 'MarkovSwitchingAR':
        """
        Entrena el modelo MS-AR.
        
        Args:
            X: Features de entrada (puede incluir la serie objetivo)
            y: Serie objetivo (opcional si está en X)
            target_column: Nombre de la columna objetivo en X
            **kwargs: Argumentos adicionales
            
        Returns:
            Self para method chaining
        """
        logger.info(f"Entrenando MS-AR con {len(X)} observaciones")
        
        # Determinar serie objetivo
        if y is not None:
            if not isinstance(y, pd.Series):
                y = pd.Series(y, index=X.index)
            self.y_training = y
        elif target_column in X.columns:
            self.y_training = X[target_column]
        else:
            raise ValueError(f"Serie objetivo no encontrada. Especificar 'y' o columna '{target_column}' en X")
        
        # Validar serie temporal
        self._validate_time_series(self.y_training)
        
        # Preparar variables exógenas (si las hay)
        exog_columns = [col for col in X.columns if col != target_column]
        if exog_columns:
            self.exog_training = X[exog_columns]
            logger.info(f"Variables exógenas: {exog_columns}")
        else:
            self.exog_training = None
            logger.info("Modelo solo con componente AR")
        
        # Guardar información de features
        self.feature_names = [target_column] + exog_columns
        
        # Verificar estacionariedad
        self._check_stationarity()
        
        # Crear y entrenar modelo
        try:
            self.model = self._create_ms_ar_model()
            
            logger.info("Iniciando entrenamiento MS-AR")
            fitted_model = self.model.fit(
                method=self.method,
                maxiter=self.maxiter,
                tol=self.tolerance,
                disp=False  # Suprimir output detallado
            )
            
            # Reemplazar modelo con versión entrenada
            self.model = fitted_model
            
            # Validar convergencia
            self._check_convergence()
            
            # Calcular métricas y diagnósticos
            self._calculate_training_metrics()
            self._run_diagnostics()
            
            self.is_fitted = True
            logger.info("Entrenamiento MS-AR completado exitosamente")
            
        except Exception as e:
            logger.error(f"Error en entrenamiento MS-AR: {e}")
            raise RuntimeError(f"Entrenamiento falló: {e}")
        
        return self
    
    def _create_ms_ar_model(self) -> MarkovAutoregression:
        """Crea instancia del modelo MS-AR."""
        
        # Configurar parámetros de switching
        if self.switching_ar and self.switching_variance:
            switching_ar = True
            switching_variance = True
        elif self.switching_ar:
            switching_ar = True
            switching_variance = False
        elif self.switching_variance:
            switching_ar = False
            switching_variance = True
        else:
            # Al menos uno debe hacer switching
            switching_ar = True
            switching_variance = False
            logger.warning("Activando switching_ar ya que ningún parámetro hace switching")
        
        model = MarkovAutoregression(
            endog=self.y_training,
            k_regimes=self.k_regimes,
            order=self.order,
            exog=self.exog_training,
            switching_ar=switching_ar,
            switching_variance=switching_variance,
            dates=self.y_training.index if hasattr(self.y_training, 'index') else None,
            freq=None  # Inferir automáticamente
        )
        
        return model
    
    def _validate_time_series(self, y: pd.Series):
        """Valida la serie temporal."""
        if len(y) < 50:
            raise ValueError(f"Serie muy corta: {len(y)} observaciones. Mínimo: 50")
        
        # Verificar NaN
        if y.isna().any():
            n_missing = y.isna().sum()
            logger.warning(f"Serie contiene {n_missing} valores faltantes")
            
            # Remover NaN del final/inicio pero no en el medio
            y_clean = y.dropna()
            if len(y_clean) < len(y) * 0.8:
                raise ValueError("Demasiados valores faltantes en la serie")
        
        # Verificar variabilidad
        if y.std() < 1e-6:
            raise ValueError("Serie temporal tiene varianza muy baja")
        
        # Verificar outliers extremos
        z_scores = np.abs((y - y.mean()) / y.std())
        n_outliers = (z_scores > 5).sum()
        if n_outliers > len(y) * 0.05:
            logger.warning(f"Serie contiene {n_outliers} outliers extremos")
    
    def _check_stationarity(self):
        """Verifica estacionariedad de la serie."""
        try:
            # Test ADF
            adf_result = adfuller(self.y_training.dropna(), autolag='AIC')
            
            self.stationarity_tests = {
                'adf_statistic': adf_result[0],
                'adf_pvalue': adf_result[1],
                'adf_critical_values': adf_result[4],
                'adf_conclusion': 'stationary' if adf_result[1] < 0.05 else 'non_stationary'
            }
            
            if adf_result[1] > 0.05:
                logger.warning(f"Serie podría ser no estacionaria (p-value ADF: {adf_result[1]:.4f})")
            else:
                logger.info(f"Serie es estacionaria (p-value ADF: {adf_result[1]:.4f})")
                
        except Exception as e:
            logger.warning(f"Error en test de estacionariedad: {e}")
            self.stationarity_tests = {'error': str(e)}
    
    def _check_convergence(self):
        """Verifica convergencia del modelo."""
        if hasattr(self.model, 'mle_retvals'):
            converged = self.model.mle_retvals.get('converged', False)
            if not converged:
                logger.warning("Modelo no convergió completamente")
            else:
                logger.info("Modelo convergió exitosamente")
        
        # Verificar log-likelihood
        if hasattr(self.model, 'llf'):
            if np.isnan(self.model.llf) or np.isinf(self.model.llf):
                raise ValueError("Log-likelihood inválido")
    
    def _calculate_training_metrics(self):
        """Calcula métricas de entrenamiento."""
        try:
            # Métricas básicas
            loglik = self.model.llf
            n_obs = self.model.nobs
            
            # Criterios de información
            aic = self.model.aic
            bic = self.model.bic
            hqic = self.model.hqic
            
            # Probabilidades de régimen
            regime_probs = self.model.smoothed_marginal_probabilities
            regime_predictions = np.argmax(regime_probs, axis=1)
            
            # Frecuencias de régimen
            regime_counts = np.bincount(regime_predictions, minlength=self.k_regimes)
            regime_frequencies = regime_counts / len(regime_predictions)
            
            # Persistencia de régimenes (duraciones promedio)
            regime_durations = self._calculate_regime_durations(regime_predictions)
            
            self.training_metrics = {
                'log_likelihood': loglik,
                'aic': aic,
                'bic': bic,
                'hqic': hqic,
                'n_observations': n_obs,
                'regime_frequencies': regime_frequencies.tolist(),
                'regime_durations': regime_durations,
                'min_regime_frequency': float(np.min(regime_frequencies))
            }
            
            # Información del modelo
            self.model_info = {
                'k_regimes': self.k_regimes,
                'order': self.order,
                'switching_ar': self.switching_ar,
                'switching_variance': self.switching_variance,
                'n_parameters': self.get_n_parameters(),
                'has_exog': self.exog_training is not None
            }
            
            # Parámetros estimados por régimen
            params_by_regime = self._extract_regime_parameters()
            self.model_info['parameters_by_regime'] = params_by_regime
            
            logger.info(f"Métricas calculadas: LogLik={loglik:.2f}, AIC={aic:.2f}, BIC={bic:.2f}")
            
        except Exception as e:
            logger.error(f"Error calculando métricas: {e}")
            self.training_metrics = {'error': str(e)}
    
    def _calculate_regime_durations(self, regime_sequence: np.ndarray) -> Dict[int, float]:
        """Calcula duración promedio de cada régimen."""
        durations = {}
        
        for regime in range(self.k_regimes):
            # Encontrar segmentos continuos del régimen
            in_regime = (regime_sequence == regime)
            transitions = np.diff(in_regime.astype(int))
            
            starts = np.where(transitions == 1)[0] + 1
            ends = np.where(transitions == -1)[0] + 1
            
            # Manejar casos especiales
            if in_regime[0]:
                starts = np.concatenate(([0], starts))
            if in_regime[-1]:
                ends = np.concatenate((ends, [len(regime_sequence)]))
            
            if len(starts) > 0 and len(ends) > 0:
                segment_lengths = ends[:len(starts)] - starts
                durations[regime] = float(np.mean(segment_lengths))
            else:
                durations[regime] = 0.0
        
        return durations
    
    def _extract_regime_parameters(self) -> Dict[int, Dict[str, Any]]:
        """Extrae parámetros estimados por régimen."""
        params_by_regime = {}
        
        try:
            for regime in range(self.k_regimes):
                regime_params = {}
                
                # Constante (si está incluida)
                if hasattr(self.model, 'params'):
                    param_names = self.model.param_names
                    params = self.model.params
                    
                    # Extraer parámetros AR para este régimen
                    ar_params = []
                    for lag in range(1, self.order + 1):
                        if self.switching_ar:
                            param_name = f'ar.L{lag}.{regime}'
                        else:
                            param_name = f'ar.L{lag}'
                        
                        if param_name in param_names:
                            param_idx = param_names.index(param_name)
                            ar_params.append(float(params[param_idx]))
                    
                    regime_params['ar_coefficients'] = ar_params
                    
                    # Varianza para este régimen
                    if self.switching_variance:
                        var_param_name = f'sigma2.{regime}'
                    else:
                        var_param_name = 'sigma2'
                    
                    if var_param_name in param_names:
                        param_idx = param_names.index(var_param_name)
                        regime_params['variance'] = float(params[param_idx])
                
                params_by_regime[regime] = regime_params
                
        except Exception as e:
            logger.warning(f"Error extrayendo parámetros por régimen: {e}")
        
        return params_by_regime
    
    def _run_diagnostics(self):
        """Ejecuta diagnósticos del modelo."""
        try:
            # Residuos
            if hasattr(self.model, 'resid'):
                residuals = self.model.resid
                
                # Estadísticas básicas de residuos
                self.residual_diagnostics = {
                    'mean': float(np.mean(residuals)),
                    'std': float(np.std(residuals)),
                    'skewness': float(stats.skew(residuals)),
                    'kurtosis': float(stats.kurtosis(residuals)),
                    'jarque_bera_stat': None,
                    'jarque_bera_pvalue': None
                }
                
                # Test Jarque-Bera para normalidad
                try:
                    from scipy import stats
                    jb_stat, jb_pvalue = stats.jarque_bera(residuals.dropna())
                    self.residual_diagnostics['jarque_bera_stat'] = float(jb_stat)
                    self.residual_diagnostics['jarque_bera_pvalue'] = float(jb_pvalue)
                except Exception:
                    pass
                
                logger.info("Diagnósticos de residuos completados")
            
        except Exception as e:
            logger.warning(f"Error en diagnósticos: {e}")
            self.residual_diagnostics = {'error': str(e)}
    
    def predict_states(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predice secuencia de estados (regímenes).
        
        Args:
            X: Features de entrada (debe incluir serie objetivo)
            
        Returns:
            Array con secuencia de regímenes
        """
        if not self.is_fitted:
            raise ValueError("Modelo debe estar entrenado")
        
        # Para MS-AR necesitamos la serie temporal completa para filtrado
        # Esto es una limitación vs HMM que puede predecir observación por observación
        
        # Por ahora, usar probabilidades suavizadas del entrenamiento
        # En implementación completa, se usaría filtro de Kalman
        regime_probs = self.model.smoothed_marginal_probabilities
        states = np.argmax(regime_probs, axis=1)
        
        # Truncar o expandir según el tamaño de X
        if len(X) != len(states):
            if len(X) < len(states):
                states = states[:len(X)]
            else:
                # Repetir último estado para observaciones adicionales
                last_state = states[-1]
                additional_states = np.full(len(X) - len(states), last_state)
                states = np.concatenate([states, additional_states])
        
        return states
    
    def predict_state_probabilities(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calcula probabilidades de régimen.
        
        Args:
            X: Features de entrada
            
        Returns:
            Array con probabilidades [n_samples, n_regimes]
        """
        if not self.is_fitted:
            raise ValueError("Modelo debe estar entrenado")
        
        # Usar probabilidades suavizadas del modelo entrenado
        regime_probs = self.model.smoothed_marginal_probabilities
        
        # Ajustar tamaño si es necesario
        if len(X) != len(regime_probs):
            if len(X) < len(regime_probs):
                regime_probs = regime_probs[:len(X)]
            else:
                # Repetir última fila para observaciones adicionales
                last_probs = regime_probs[-1:].reshape(1, -1)
                additional_probs = np.repeat(last_probs, len(X) - len(regime_probs), axis=0)
                regime_probs = np.vstack([regime_probs, additional_probs])
        
        return regime_probs
    
    def sample(self, n_samples: int, random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genera muestras del modelo.
        
        Args:
            n_samples: Número de muestras
            random_state: Semilla aleatoria
            
        Returns:
            Tupla (observaciones, regímenes)
        """
        if not self.is_fitted:
            raise ValueError("Modelo debe estar entrenado")
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Simular del modelo
        try:
            simulation = self.model.simulate(
                n_samples, 
                regime_transition=None  # Usar matriz de transición estimada
            )
            
            simulated_data = simulation['endog']
            simulated_regimes = simulation['regime']
            
            return simulated_data, simulated_regimes
            
        except Exception as e:
            logger.error(f"Error en simulación: {e}")
            # Fallback: generar datos dummy
            dummy_data = np.random.normal(0, 1, n_samples)
            dummy_regimes = np.random.randint(0, self.k_regimes, n_samples)
            return dummy_data, dummy_regimes
    
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
        
        # Para MS-AR, el score es la log-likelihood del entrenamiento
        # En implementación completa, se recalcularía para nuevos datos
        return self.model.llf
    
    def get_n_parameters(self) -> int:
        """
        Calcula número de parámetros.
        
        Returns:
            Número de parámetros
        """
        if not self.is_fitted:
            # Estimación antes del entrenamiento
            n_ar_params = self.order * (self.k_regimes if self.switching_ar else 1)
            n_var_params = self.k_regimes if self.switching_variance else 1
            n_transition_params = self.k_regimes * (self.k_regimes - 1)
            
            # Exógenas (si las hay)
            n_exog_params = 0
            if self.exog_training is not None:
                n_exog_vars = self.exog_training.shape[1]
                n_exog_params = n_exog_vars * self.k_regimes
            
            return n_ar_params + n_var_params + n_transition_params + n_exog_params
        
        # Después del entrenamiento
        if hasattr(self.model, 'params'):
            return len(self.model.params)
        
        return 0
    
    def get_transition_matrix(self) -> Optional[np.ndarray]:
        """
        Obtiene matriz de transición de regímenes.
        
        Returns:
            Matriz de transición [k_regimes, k_regimes]
        """
        if not self.is_fitted:
            return None
        
        if hasattr(self.model, 'transition_matrix'):
            return self.model.transition_matrix
        
        return None
    
    def forecast(self, steps: int, exog_future: Optional[pd.DataFrame] = None) -> Dict[str, np.ndarray]:
        """
        Genera pronósticos fuera de muestra.
        
        Args:
            steps: Número de períodos a pronosticar
            exog_future: Variables exógenas futuras (si las hay)
            
        Returns:
            Dict con pronósticos y intervalos de confianza
        """
        if not self.is_fitted:
            raise ValueError("Modelo debe estar entrenado")
        
        try:
            # Generar pronósticos
            forecast_result = self.model.forecast(steps=steps, exog=exog_future)
            
            forecast_dict = {
                'forecast': forecast_result.forecast,
                'forecast_se': forecast_result.forecast_se if hasattr(forecast_result, 'forecast_se') else None,
                'conf_int': forecast_result.conf_int() if hasattr(forecast_result, 'conf_int') else None
            }
            
            # Agregar pronósticos de régimen si están disponibles
            if hasattr(forecast_result, 'regime_forecast'):
                forecast_dict['regime_forecast'] = forecast_result.regime_forecast
            
            return forecast_dict
            
        except Exception as e:
            logger.error(f"Error en pronóstico: {e}")
            # Fallback: pronóstico simple
            last_value = self.y_training.iloc[-1] if hasattr(self.y_training, 'iloc') else 0
            return {
                'forecast': np.full(steps, last_value),
                'forecast_se': None,
                'conf_int': None,
                'regime_forecast': None
            }
    
    def interpret_regimes(self) -> Dict[int, Dict[str, Any]]:
        """
        Interpreta los regímenes del modelo.
        
        Returns:
            Dict con interpretación de cada régimen
        """
        if not self.is_fitted:
            raise ValueError("Modelo debe estar entrenado")
        
        regime_interpretation = {}
        params_by_regime = self.model_info.get('parameters_by_regime', {})
        
        for regime in range(self.k_regimes):
            regime_params = params_by_regime.get(regime, {})
            
            # Analizar coeficientes AR
            ar_coeffs = regime_params.get('ar_coefficients', [])
            variance = regime_params.get('variance', 0)
            
            # Calcular persistencia (suma de coeficientes AR)
            persistence = sum(ar_coeffs) if ar_coeffs else 0
            
            # Clasificar régimen
            if persistence > 0.8:
                regime_type = "altamente_persistente"
            elif persistence > 0.3:
                regime_type = "moderadamente_persistente"
            elif persistence > -0.3:
                regime_type = "mean_reverting"
            else:
                regime_type = "fuertemente_mean_reverting"
            
            volatility_level = "alta" if variance > np.median([
                params_by_regime.get(r, {}).get('variance', 0) 
                for r in range(self.k_regimes)
            ]) else "baja"
            
            regime_interpretation[regime] = {
                'ar_coefficients': ar_coeffs,
                'variance': variance,
                'persistence': persistence,
                'regime_type': regime_type,
                'volatility_level': volatility_level,
                'duration': self.training_metrics.get('regime_durations', {}).get(regime, 0),
                'frequency': self.training_metrics.get('regime_frequencies', [0] * self.k_regimes)[regime]
            }
        
        return regime_interpretation


if __name__ == "__main__":
    # Ejemplo de uso
    try:
        from ..config_loader import load_config
        
        config = load_config("config/binance_spot_example.yaml")
        
        # Crear serie temporal con cambios de régimen
        np.random.seed(42)
        n_samples = 300
        
        # Simular 2 regímenes con diferentes comportamientos AR
        regime_changes = [0, 150, 300]
        data = []
        regimes_true = []
        
        for i, (start, end) in enumerate(zip(regime_changes[:-1], regime_changes[1:])):
            regime_size = end - start
            
            if i == 0:  # Régimen 1: AR(1) con coef 0.3, baja varianza
                ar_coeff = 0.3
                variance = 0.01
            else:  # Régimen 2: AR(1) con coef -0.2, alta varianza
                ar_coeff = -0.2
                variance = 0.04
            
            # Generar serie AR(1)
            regime_data = [0]  # Valor inicial
            for t in range(1, regime_size):
                next_val = ar_coeff * regime_data[t-1] + np.random.normal(0, np.sqrt(variance))
                regime_data.append(next_val)
            
            data.extend(regime_data)
            regimes_true.extend([i] * regime_size)
        
        # Crear DataFrame
        y_series = pd.Series(data, name='log_return_1')
        X = pd.DataFrame({'log_return_1': y_series})
        
        print(f"Serie temporal creada: {len(X)} observaciones")
        print(f"Estadísticas: media={y_series.mean():.4f}, std={y_series.std():.4f}")
        
        # Entrenar modelo
        model = MarkovSwitchingAR(config, k_regimes=2, order=1)
        model.fit(X, target_column='log_return_1')
        
        print(f"\nModelo MS-AR entrenado exitosamente")
        
        # Métricas
        print(f"Métricas del modelo:")
        for metric, value in model.training_metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
        
        # Predecir regímenes
        predicted_regimes = model.predict_states(X)
        
        # Comparar con regímenes verdaderos
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(regimes_true, predicted_regimes)
        print(f"Adjusted Rand Index: {ari:.3f}")
        
        # Interpretación de regímenes
        interpretation = model.interpret_regimes()
        print(f"\nInterpretación de regímenes:")
        for regime, info in interpretation.items():
            print(f"  Régimen {regime}:")
            print(f"    Tipo: {info['regime_type']}")
            print(f"    Persistencia: {info['persistence']:.3f}")
            print(f"    Volatilidad: {info['volatility_level']}")
            print(f"    Duración promedio: {info['duration']:.1f} períodos")
        
        # Test de pronóstico
        forecast = model.forecast(steps=5)
        print(f"\nPronóstico 5 períodos adelante:")
        print(forecast['forecast'])
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
