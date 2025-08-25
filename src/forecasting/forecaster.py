"""
Sistema de pronóstico probabilístico para markov-crypto-forecaster.

Genera predicciones probabilísticas usando modelos de Markov entrenados,
incluyendo inferencia de régimen, probabilidades futuras y escenarios.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import warnings

from ..models.base import MarkovModelBase
from ..config_loader import Config
from ..logging_utils import get_logger, log_execution_time
from ..utils_time import timeframe_to_minutes

logger = get_logger(__name__)


class MarkovForecaster:
    """
    Sistema de pronóstico probabilístico usando modelos de Markov.
    
    Características:
    - Inferencia de régimen actual
    - Probabilidades de transición futuras
    - Distribuciones predictivas por régimen
    - Escenarios probabilísticos (p10, p50, p90)
    - Ensembles de múltiples modelos
    - Intervalos de confianza
    """
    
    def __init__(self, config: Union[Config, dict]):
        """
        Inicializa el sistema de pronóstico.
        
        Args:
            config: Configuración del sistema
        """
        if isinstance(config, dict):
            self.config = config
        else:
            self.config = config.dict()
            
        self.forecasting_config = self.config.get('forecasting', {})
        self.paths_config = self.config.get('paths', {})
        
        # Configuración de pronóstico
        self.horizons = self.forecasting_config.get('horizons', [1, 3, 5])
        self.confidence_levels = self.forecasting_config.get('confidence_levels', [0.1, 0.5, 0.9])
        self.ensemble_method = self.forecasting_config.get('ensemble_method', 'average')
        
        # Modelos cargados
        self.models = {}
        self.ensemble_weights = {}
        
        # Cache de predicciones
        self.prediction_cache = {}
        self.last_prediction_time = None
        
        # Crear directorio de señales
        self.signals_dir = Path(self.paths_config.get('signals_dir', 'data/signals'))
        self.signals_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Forecaster inicializado: horizontes={self.horizons}, método ensemble={self.ensemble_method}")
    
    def add_model(self, model: MarkovModelBase, name: str, weight: float = 1.0):
        """
        Agrega un modelo al ensemble.
        
        Args:
            model: Modelo entrenado
            name: Nombre del modelo
            weight: Peso en el ensemble
        """
        if not model.is_fitted:
            raise ValueError(f"Modelo '{name}' debe estar entrenado")
        
        self.models[name] = model
        self.ensemble_weights[name] = weight
        
        logger.info(f"Modelo '{name}' agregado al ensemble (peso={weight})")
    
    def remove_model(self, name: str):
        """
        Remueve un modelo del ensemble.
        
        Args:
            name: Nombre del modelo a remover
        """
        if name in self.models:
            del self.models[name]
            del self.ensemble_weights[name]
            logger.info(f"Modelo '{name}' removido del ensemble")
    
    def clear_models(self):
        """Limpia todos los modelos del ensemble."""
        self.models.clear()
        self.ensemble_weights.clear()
        self.prediction_cache.clear()
        logger.info("Todos los modelos removidos del ensemble")
    
    @log_execution_time("generate_forecast")
    def generate_forecast(self, X: pd.DataFrame, 
                         horizons: Optional[List[int]] = None,
                         include_regime_forecast: bool = True,
                         include_scenarios: bool = True,
                         use_cache: bool = True) -> Dict[str, Any]:
        """
        Genera pronóstico probabilístico completo.
        
        Args:
            X: Features actuales
            horizons: Horizontes de pronóstico (usa config si None)
            include_regime_forecast: Si incluir pronóstico de régimen
            include_scenarios: Si incluir escenarios probabilísticos
            use_cache: Si usar cache de predicciones
            
        Returns:
            Dict con pronósticos y métricas
            
        Raises:
            ValueError: Si no hay modelos cargados
        """
        if not self.models:
            raise ValueError("No hay modelos cargados para pronóstico")
        
        if horizons is None:
            horizons = self.horizons
        
        # Verificar cache
        cache_key = self._generate_cache_key(X, horizons)
        if use_cache and cache_key in self.prediction_cache:
            logger.info("Usando pronóstico desde cache")
            return self.prediction_cache[cache_key]
        
        logger.info(f"Generando pronóstico para {len(X)} observaciones, horizontes={horizons}")
        
        # Inferir régimen actual
        current_regime_info = self._infer_current_regime(X)
        
        # Generar pronósticos por modelo
        model_forecasts = {}
        for model_name, model in self.models.items():
            try:
                model_forecast = self._forecast_single_model(model, X, horizons)
                model_forecasts[model_name] = model_forecast
            except Exception as e:
                logger.error(f"Error en pronóstico de modelo '{model_name}': {e}")
                continue
        
        if not model_forecasts:
            raise ValueError("No se pudieron generar pronósticos de ningún modelo")
        
        # Crear ensemble
        ensemble_forecast = self._create_ensemble_forecast(model_forecasts, horizons)
        
        # Construir resultado final
        forecast_result = {
            'timestamp': datetime.now().isoformat(),
            'current_regime': current_regime_info,
            'ensemble_forecast': ensemble_forecast,
            'model_forecasts': model_forecasts,
            'horizons': horizons,
            'n_models': len(model_forecasts),
            'ensemble_method': self.ensemble_method
        }
        
        # Pronósticos de régimen
        if include_regime_forecast:
            regime_forecasts = self._forecast_regime_evolution(X, horizons)
            forecast_result['regime_forecasts'] = regime_forecasts
        
        # Escenarios probabilísticos
        if include_scenarios:
            scenarios = self._generate_probability_scenarios(ensemble_forecast, horizons)
            forecast_result['scenarios'] = scenarios
        
        # Métricas de confianza
        confidence_metrics = self._calculate_forecast_confidence(model_forecasts)
        forecast_result['confidence_metrics'] = confidence_metrics
        
        # Cache resultado
        if use_cache:
            self.prediction_cache[cache_key] = forecast_result
            self.last_prediction_time = datetime.now()
        
        logger.info("Pronóstico generado exitosamente")
        return forecast_result
    
    def _infer_current_regime(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Infiere el régimen actual basado en observaciones recientes.
        
        Args:
            X: Features actuales
            
        Returns:
            Información del régimen actual
        """
        regime_probs_by_model = {}
        regime_predictions_by_model = {}
        
        for model_name, model in self.models.items():
            try:
                # Usar últimas observaciones para inferencia
                recent_X = X.tail(min(50, len(X)))
                
                # Probabilidades de régimen
                state_probs = model.predict_state_probabilities(recent_X)
                current_probs = state_probs[-1]  # Última observación
                
                # Predicción de régimen más probable
                current_state = np.argmax(current_probs)
                
                regime_probs_by_model[model_name] = current_probs.tolist()
                regime_predictions_by_model[model_name] = int(current_state)
                
            except Exception as e:
                logger.warning(f"Error infiriendo régimen con modelo '{model_name}': {e}")
                continue
        
        if not regime_probs_by_model:
            return {'error': 'No se pudo inferir régimen actual'}
        
        # Crear ensemble de probabilidades de régimen
        ensemble_probs = self._ensemble_regime_probabilities(regime_probs_by_model)
        ensemble_regime = int(np.argmax(ensemble_probs))
        
        # Calcular confianza (entropía inversa)
        entropy = -np.sum(ensemble_probs * np.log(ensemble_probs + 1e-10))
        max_entropy = np.log(len(ensemble_probs))
        confidence = 1 - (entropy / max_entropy)
        
        return {
            'ensemble_regime': ensemble_regime,
            'ensemble_probabilities': ensemble_probs.tolist(),
            'confidence': float(confidence),
            'entropy': float(entropy),
            'model_predictions': regime_predictions_by_model,
            'model_probabilities': regime_probs_by_model,
            'n_models_consensus': len(regime_probs_by_model)
        }
    
    def _forecast_single_model(self, model: MarkovModelBase, 
                              X: pd.DataFrame, 
                              horizons: List[int]) -> Dict[str, Any]:
        """
        Genera pronóstico para un modelo individual.
        
        Args:
            model: Modelo a usar para pronóstico
            X: Features actuales
            horizons: Horizontes de pronóstico
            
        Returns:
            Pronóstico del modelo
        """
        forecast = {
            'model_type': model.model_type,
            'n_states': model.n_states,
            'horizons': {}
        }
        
        # Estado actual
        current_states = model.predict_states(X.tail(10))
        current_state = current_states[-1]
        current_probs = model.predict_state_probabilities(X.tail(1))[0]
        
        # Obtener parámetros del modelo
        transition_matrix = model.get_transition_matrix()
        emission_params = model.get_emission_parameters()
        
        for horizon in horizons:
            horizon_forecast = {
                'horizon': horizon,
                'state_predictions': {},
                'return_predictions': {},
                'confidence_intervals': {}
            }
            
            # Pronóstico de probabilidades de estado
            if transition_matrix is not None:
                future_state_probs = self._evolve_state_probabilities(
                    current_probs, transition_matrix, horizon
                )
                horizon_forecast['state_predictions'] = {
                    'probabilities': future_state_probs.tolist(),
                    'most_likely_state': int(np.argmax(future_state_probs))
                }
            
            # Pronóstico de retornos (si es posible)
            return_forecast = self._forecast_returns(
                model, current_state, future_state_probs if transition_matrix is not None else current_probs, 
                emission_params, horizon
            )
            horizon_forecast['return_predictions'] = return_forecast
            
            # Intervalos de confianza
            confidence_intervals = self._calculate_confidence_intervals(
                return_forecast, self.confidence_levels
            )
            horizon_forecast['confidence_intervals'] = confidence_intervals
            
            forecast['horizons'][horizon] = horizon_forecast
        
        return forecast
    
    def _evolve_state_probabilities(self, initial_probs: np.ndarray, 
                                   transition_matrix: np.ndarray, 
                                   steps: int) -> np.ndarray:
        """
        Evoluciona probabilidades de estado usando matriz de transición.
        
        Args:
            initial_probs: Probabilidades iniciales
            transition_matrix: Matriz de transición
            steps: Número de pasos hacia adelante
            
        Returns:
            Probabilidades evolucionadas
        """
        # P(s_t+k) = P(s_t) * T^k
        evolved_probs = initial_probs.copy()
        
        for _ in range(steps):
            evolved_probs = evolved_probs @ transition_matrix
        
        return evolved_probs
    
    def _forecast_returns(self, model: MarkovModelBase, 
                         current_state: int,
                         future_state_probs: np.ndarray,
                         emission_params: Optional[Dict[str, Any]],
                         horizon: int) -> Dict[str, Any]:
        """
        Genera pronóstico de retornos basado en régimen.
        
        Args:
            model: Modelo para pronóstico
            current_state: Estado actual
            future_state_probs: Probabilidades futuras de estado
            emission_params: Parámetros de emisión
            horizon: Horizonte de pronóstico
            
        Returns:
            Pronóstico de retornos
        """
        return_forecast = {
            'point_forecast': 0.0,
            'variance_forecast': 0.0,
            'distribution_type': 'unknown'
        }
        
        try:
            if model.model_type == 'hmm_gaussian' and emission_params:
                # Para HMM Gaussiano, usar medias y covarianzas
                means = np.array(emission_params['means'])
                covariances = emission_params['covariances']
                
                # Retorno esperado ponderado por probabilidades de estado
                expected_return = np.sum(future_state_probs * means[:, 0])  # Asumir primera feature es retorno
                
                # Varianza esperada
                expected_variance = 0.0
                for state in range(len(future_state_probs)):
                    prob = future_state_probs[state]
                    if emission_params['covariance_type'] == 'full':
                        state_var = covariances[state][0, 0]  # Varianza del retorno
                    elif emission_params['covariance_type'] == 'diag':
                        state_var = covariances[state][0]
                    elif emission_params['covariance_type'] == 'spherical':
                        state_var = covariances[state]
                    else:  # tied
                        state_var = covariances[0, 0]
                    
                    expected_variance += prob * (state_var + (means[state, 0] - expected_return)**2)
                
                return_forecast.update({
                    'point_forecast': float(expected_return * horizon),  # Escalar por horizonte
                    'variance_forecast': float(expected_variance * horizon),
                    'distribution_type': 'gaussian_mixture'
                })
            
            elif model.model_type == 'ms_ar':
                # Para MS-AR, usar parámetros autorregresivos
                # Esto requeriría implementación más compleja con simulación
                # Por ahora, usar aproximación simple
                last_return = 0.0  # Se obtendría de los datos
                
                return_forecast.update({
                    'point_forecast': float(last_return * 0.1),  # Placeholder
                    'variance_forecast': 0.01,
                    'distribution_type': 'regime_switching_ar'
                })
            
            elif model.model_type == 'hmm_discrete':
                # Para HMM discreto, mapear símbolos a retornos aproximados
                # Requiere mapeo inverso de discretización
                return_forecast.update({
                    'point_forecast': 0.0,
                    'variance_forecast': 0.01,
                    'distribution_type': 'discrete_mixture'
                })
        
        except Exception as e:
            logger.warning(f"Error en pronóstico de retornos: {e}")
        
        return return_forecast
    
    def _calculate_confidence_intervals(self, return_forecast: Dict[str, Any], 
                                      confidence_levels: List[float]) -> Dict[str, Dict[str, float]]:
        """
        Calcula intervalos de confianza para pronósticos.
        
        Args:
            return_forecast: Pronóstico de retornos
            confidence_levels: Niveles de confianza
            
        Returns:
            Intervalos de confianza
        """
        intervals = {}
        
        point_forecast = return_forecast['point_forecast']
        variance_forecast = return_forecast['variance_forecast']
        std_forecast = np.sqrt(variance_forecast)
        
        for level in confidence_levels:
            # Usar distribución normal para intervalos
            from scipy import stats
            
            alpha = 1 - level
            z_score = stats.norm.ppf(1 - alpha/2)
            
            lower = point_forecast - z_score * std_forecast
            upper = point_forecast + z_score * std_forecast
            
            intervals[f'p{int(level*100)}'] = {
                'lower': float(lower),
                'upper': float(upper),
                'width': float(upper - lower)
            }
        
        return intervals
    
    def _create_ensemble_forecast(self, model_forecasts: Dict[str, Dict[str, Any]], 
                                 horizons: List[int]) -> Dict[str, Any]:
        """
        Crea ensemble de pronósticos de múltiples modelos.
        
        Args:
            model_forecasts: Pronósticos por modelo
            horizons: Horizontes de pronóstico
            
        Returns:
            Pronóstico ensemble
        """
        ensemble = {'horizons': {}}
        
        # Normalizar pesos
        total_weight = sum(self.ensemble_weights.values())
        normalized_weights = {
            name: weight / total_weight 
            for name, weight in self.ensemble_weights.items()
            if name in model_forecasts
        }
        
        for horizon in horizons:
            horizon_ensemble = {
                'horizon': horizon,
                'point_forecast': 0.0,
                'variance_forecast': 0.0,
                'state_probabilities': None,
                'confidence_intervals': {}
            }
            
            # Agregar pronósticos ponderados
            total_point = 0.0
            total_variance = 0.0
            state_probs_sum = None
            
            for model_name, forecast in model_forecasts.items():
                if model_name not in normalized_weights:
                    continue
                
                weight = normalized_weights[model_name]
                horizon_data = forecast['horizons'].get(horizon, {})
                
                # Retornos
                return_pred = horizon_data.get('return_predictions', {})
                point = return_pred.get('point_forecast', 0.0)
                variance = return_pred.get('variance_forecast', 0.0)
                
                total_point += weight * point
                total_variance += weight * variance  # Aproximación simple
                
                # Estados (promedio ponderado de probabilidades)
                state_pred = horizon_data.get('state_predictions', {})
                if 'probabilities' in state_pred:
                    state_probs = np.array(state_pred['probabilities'])
                    if state_probs_sum is None:
                        state_probs_sum = weight * state_probs
                    else:
                        # Ensure same length
                        if len(state_probs) == len(state_probs_sum):
                            state_probs_sum += weight * state_probs
            
            horizon_ensemble['point_forecast'] = total_point
            horizon_ensemble['variance_forecast'] = total_variance
            
            if state_probs_sum is not None:
                horizon_ensemble['state_probabilities'] = state_probs_sum.tolist()
                horizon_ensemble['most_likely_state'] = int(np.argmax(state_probs_sum))
            
            # Intervalos de confianza ensemble
            ensemble_return_forecast = {
                'point_forecast': total_point,
                'variance_forecast': total_variance
            }
            confidence_intervals = self._calculate_confidence_intervals(
                ensemble_return_forecast, self.confidence_levels
            )
            horizon_ensemble['confidence_intervals'] = confidence_intervals
            
            ensemble['horizons'][horizon] = horizon_ensemble
        
        ensemble['method'] = self.ensemble_method
        ensemble['weights'] = normalized_weights
        
        return ensemble
    
    def _ensemble_regime_probabilities(self, regime_probs_by_model: Dict[str, List[float]]) -> np.ndarray:
        """
        Crea ensemble de probabilidades de régimen.
        
        Args:
            regime_probs_by_model: Probabilidades por modelo
            
        Returns:
            Probabilidades ensemble
        """
        # Determinar número máximo de estados
        max_states = max(len(probs) for probs in regime_probs_by_model.values())
        
        # Normalizar pesos para modelos válidos
        valid_models = list(regime_probs_by_model.keys())
        total_weight = sum(self.ensemble_weights.get(name, 1.0) for name in valid_models)
        
        ensemble_probs = np.zeros(max_states)
        
        for model_name, probs in regime_probs_by_model.items():
            weight = self.ensemble_weights.get(model_name, 1.0) / total_weight
            probs_array = np.array(probs)
            
            # Pad with zeros if necessary
            if len(probs_array) < max_states:
                padded_probs = np.zeros(max_states)
                padded_probs[:len(probs_array)] = probs_array
                probs_array = padded_probs
            
            ensemble_probs += weight * probs_array[:max_states]
        
        # Renormalizar
        ensemble_probs = ensemble_probs / ensemble_probs.sum()
        
        return ensemble_probs
    
    def _forecast_regime_evolution(self, X: pd.DataFrame, 
                                  horizons: List[int]) -> Dict[str, Any]:
        """
        Pronóstica evolución de regímenes.
        
        Args:
            X: Features actuales
            horizons: Horizontes de pronóstico
            
        Returns:
            Evolución de regímenes pronosticada
        """
        regime_evolution = {}
        
        for model_name, model in self.models.items():
            try:
                transition_matrix = model.get_transition_matrix()
                if transition_matrix is None:
                    continue
                
                # Estado actual
                current_probs = model.predict_state_probabilities(X.tail(1))[0]
                
                model_evolution = {}
                for horizon in horizons:
                    future_probs = self._evolve_state_probabilities(
                        current_probs, transition_matrix, horizon
                    )
                    
                    model_evolution[horizon] = {
                        'probabilities': future_probs.tolist(),
                        'most_likely_state': int(np.argmax(future_probs)),
                        'entropy': float(-np.sum(future_probs * np.log(future_probs + 1e-10)))
                    }
                
                regime_evolution[model_name] = model_evolution
                
            except Exception as e:
                logger.warning(f"Error en evolución de régimen para {model_name}: {e}")
                continue
        
        return regime_evolution
    
    def _generate_probability_scenarios(self, ensemble_forecast: Dict[str, Any], 
                                       horizons: List[int]) -> Dict[str, Any]:
        """
        Genera escenarios probabilísticos.
        
        Args:
            ensemble_forecast: Pronóstico ensemble
            horizons: Horizontes de pronóstico
            
        Returns:
            Escenarios probabilísticos
        """
        scenarios = {}
        
        scenario_levels = [0.1, 0.25, 0.5, 0.75, 0.9]  # Percentiles
        
        for horizon in horizons:
            horizon_data = ensemble_forecast['horizons'].get(horizon, {})
            point_forecast = horizon_data.get('point_forecast', 0.0)
            variance_forecast = horizon_data.get('variance_forecast', 0.01)
            
            # Generar escenarios usando distribución normal
            std_forecast = np.sqrt(variance_forecast)
            
            from scipy import stats
            
            scenario_values = {}
            for level in scenario_levels:
                value = stats.norm.ppf(level, loc=point_forecast, scale=std_forecast)
                scenario_values[f'p{int(level*100)}'] = float(value)
            
            scenarios[horizon] = {
                'point_forecast': point_forecast,
                'scenarios': scenario_values,
                'scenario_range': float(scenario_values['p90'] - scenario_values['p10']),
                'upside_potential': float(scenario_values['p90'] - point_forecast),
                'downside_risk': float(point_forecast - scenario_values['p10'])
            }
        
        return scenarios
    
    def _calculate_forecast_confidence(self, model_forecasts: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calcula métricas de confianza del pronóstico.
        
        Args:
            model_forecasts: Pronósticos por modelo
            
        Returns:
            Métricas de confianza
        """
        confidence_metrics = {
            'n_models': len(model_forecasts),
            'model_agreement': {},
            'forecast_uncertainty': {},
            'ensemble_stability': 0.0
        }
        
        # Analizar acuerdo entre modelos
        for horizon in self.horizons:
            point_forecasts = []
            state_predictions = []
            
            for forecast in model_forecasts.values():
                horizon_data = forecast['horizons'].get(horizon, {})
                
                # Retornos
                return_pred = horizon_data.get('return_predictions', {})
                if 'point_forecast' in return_pred:
                    point_forecasts.append(return_pred['point_forecast'])
                
                # Estados
                state_pred = horizon_data.get('state_predictions', {})
                if 'most_likely_state' in state_pred:
                    state_predictions.append(state_pred['most_likely_state'])
            
            # Métricas de acuerdo
            horizon_agreement = {}
            
            if len(point_forecasts) > 1:
                # Dispersión de pronósticos de retorno
                forecast_std = np.std(point_forecasts)
                forecast_range = np.max(point_forecasts) - np.min(point_forecasts)
                
                horizon_agreement['return_std'] = float(forecast_std)
                horizon_agreement['return_range'] = float(forecast_range)
                horizon_agreement['return_agreement'] = float(1.0 / (1.0 + forecast_std))
            
            if len(state_predictions) > 1:
                # Acuerdo en predicción de estado
                unique_states, counts = np.unique(state_predictions, return_counts=True)
                max_agreement = np.max(counts) / len(state_predictions)
                
                horizon_agreement['state_agreement'] = float(max_agreement)
                horizon_agreement['state_consensus'] = len(unique_states) == 1
            
            confidence_metrics['model_agreement'][horizon] = horizon_agreement
        
        # Calcular estabilidad general del ensemble
        if model_forecasts:
            # Usar variación en el primer horizonte como proxy
            first_horizon = min(self.horizons)
            first_agreement = confidence_metrics['model_agreement'].get(first_horizon, {})
            return_agreement = first_agreement.get('return_agreement', 0.5)
            state_agreement = first_agreement.get('state_agreement', 0.5)
            
            confidence_metrics['ensemble_stability'] = float((return_agreement + state_agreement) / 2)
        
        return confidence_metrics
    
    def _generate_cache_key(self, X: pd.DataFrame, horizons: List[int]) -> str:
        """
        Genera clave para cache de predicciones.
        
        Args:
            X: Features
            horizons: Horizontes
            
        Returns:
            Clave de cache
        """
        # Usar hash de últimas observaciones y configuración
        last_obs = X.tail(5).values.flatten()
        data_hash = hash(tuple(last_obs.round(6)))  # Redondear para estabilidad
        config_hash = hash(tuple(sorted(horizons)))
        models_hash = hash(tuple(sorted(self.models.keys())))
        
        return f"{data_hash}_{config_hash}_{models_hash}"
    
    def save_forecast(self, forecast_result: Dict[str, Any], 
                     symbol: str = "unknown",
                     suffix: str = "") -> Path:
        """
        Guarda pronóstico en archivo.
        
        Args:
            forecast_result: Resultado del pronóstico
            symbol: Símbolo del activo
            suffix: Sufijo para el archivo
            
        Returns:
            Path del archivo guardado
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"forecast_{symbol}_{timestamp}"
        if suffix:
            filename += f"_{suffix}"
        filename += ".json"
        
        filepath = self.signals_dir / filename
        
        # Preparar datos para JSON
        import json
        
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            return obj
        
        json_data = json.loads(json.dumps(forecast_result, default=convert_numpy))
        
        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        logger.info(f"Pronóstico guardado: {filepath}")
        return filepath
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas resumen del forecaster.
        
        Returns:
            Estadísticas del sistema
        """
        return {
            'n_models_loaded': len(self.models),
            'model_types': [model.model_type for model in self.models.values()],
            'ensemble_weights': self.ensemble_weights.copy(),
            'horizons': self.horizons,
            'confidence_levels': self.confidence_levels,
            'ensemble_method': self.ensemble_method,
            'cache_size': len(self.prediction_cache),
            'last_prediction': self.last_prediction_time.isoformat() if self.last_prediction_time else None
        }


if __name__ == "__main__":
    # Ejemplo de uso
    try:
        from ..config_loader import load_config
        from ..models.hmm_gaussian import GaussianHMM
        from ..models.hmm_discrete import DiscreteHMM
        
        config = load_config("config/binance_spot_example.yaml")
        
        # Crear datos de ejemplo
        np.random.seed(42)
        n_samples = 100
        
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='1H')
        returns = np.random.normal(0, 0.02, n_samples)
        volatility = np.random.normal(0.02, 0.005, n_samples)
        volume = np.random.lognormal(0, 0.3, n_samples)
        
        X = pd.DataFrame({
            'log_return_1': returns,
            'volatility': volatility,
            'volume_zscore': volume
        }, index=dates)
        
        print(f"Datos de ejemplo: {len(X)} observaciones")
        
        # Entrenar modelos
        hmm_gaussian = GaussianHMM(config, n_components=3)
        hmm_gaussian.fit(X)
        
        hmm_discrete = DiscreteHMM(config, n_components=3) 
        hmm_discrete.fit(X)
        
        # Crear forecaster
        forecaster = MarkovForecaster(config)
        forecaster.add_model(hmm_gaussian, 'hmm_gaussian', weight=0.6)
        forecaster.add_model(hmm_discrete, 'hmm_discrete', weight=0.4)
        
        print(f"Forecaster configurado con {len(forecaster.models)} modelos")
        
        # Generar pronóstico
        forecast = forecaster.generate_forecast(X, horizons=[1, 3, 5])
        
        print(f"\nPronóstico generado:")
        print(f"Régimen actual (ensemble): {forecast['current_regime']['ensemble_regime']}")
        print(f"Confianza: {forecast['current_regime']['confidence']:.3f}")
        
        # Mostrar pronósticos por horizonte
        ensemble_forecast = forecast['ensemble_forecast']
        for horizon in [1, 3, 5]:
            horizon_data = ensemble_forecast['horizons'][horizon]
            point = horizon_data['point_forecast']
            print(f"Horizonte {horizon}: retorno esperado = {point:.4f}")
        
        # Escenarios
        scenarios = forecast['scenarios']
        print(f"\nEscenarios para horizonte 1:")
        h1_scenarios = scenarios[1]['scenarios']
        print(f"  P10: {h1_scenarios['p10']:.4f}")
        print(f"  P50: {h1_scenarios['p50']:.4f}")
        print(f"  P90: {h1_scenarios['p90']:.4f}")
        
        # Guardar pronóstico
        saved_path = forecaster.save_forecast(forecast, "BTC_USDT")
        print(f"\nPronóstico guardado: {saved_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
