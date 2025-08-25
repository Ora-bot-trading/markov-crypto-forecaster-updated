"""
Lógica de posiciones y señales de trading para markov-crypto-forecaster.

Mapea regímenes de Markov a señales de trading, aplica filtros
y calcula position sizing basado en probabilidades de régimen.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from datetime import datetime
from enum import Enum

from ..config_loader import Config
from ..logging_utils import get_logger, log_execution_time

logger = get_logger(__name__)


class SignalType(Enum):
    """Tipos de señales de trading."""
    LONG = 1
    SHORT = -1
    FLAT = 0
    HOLD = 0


class SignalStrength(Enum):
    """Fuerza de las señales."""
    WEAK = 0.25
    MODERATE = 0.5
    STRONG = 0.75
    VERY_STRONG = 1.0


class PositionLogic:
    """
    Lógica de posiciones basada en regímenes de Markov.
    
    Convierte probabilidades de régimen en señales de trading,
    aplica filtros de mercado y calcula position sizing.
    
    Características:
    - Mapeo configurable de régimen a señal
    - Filtros de volatilidad y momentum
    - Position sizing basado en Kelly fraction
    - Manejo de incertidumbre (régimenes mixtos)
    - Señales graduales basadas en probabilidades
    """
    
    def __init__(self, config: Union[Config, dict]):
        """
        Inicializa la lógica de posiciones.
        
        Args:
            config: Configuración del sistema
        """
        if isinstance(config, dict):
            self.config = config
        else:
            self.config = config.dict()
            
        self.strategy_config = self.config.get('strategy', {})
        
        # Mapeo de régimen a señal
        self.regime_mapping = self.strategy_config.get('regime_mapping', {
            0: 'flat',
            1: 'long', 
            2: 'short'
        })
        
        # Filtros de señales
        self.filters_config = self.strategy_config.get('filters', {})
        self.min_regime_prob = self.filters_config.get('min_regime_prob', 0.6)
        self.vol_filter = self.filters_config.get('vol_filter', True)
        self.vol_threshold = self.filters_config.get('vol_threshold', 2.0)
        
        # Position sizing
        self.sizing_config = self.strategy_config.get('sizing', {})
        self.sizing_method = self.sizing_config.get('method', 'kelly_fraction')
        self.kelly_fraction_limit = self.sizing_config.get('kelly_fraction_limit', 0.3)
        self.vol_target = self.sizing_config.get('vol_target', 0.15)
        self.max_leverage = self.sizing_config.get('max_leverage', 1.0)
        
        # Estado interno
        self.last_signals = {}
        self.signal_history = []
        
        logger.info(f"PositionLogic inicializada: mapeo={self.regime_mapping}, "
                   f"sizing={self.sizing_method}")
    
    @log_execution_time("generate_signals")
    def generate_signals(self, regime_forecast: Dict[str, Any],
                        market_data: pd.DataFrame,
                        features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Genera señales de trading basadas en pronóstico de régimen.
        
        Args:
            regime_forecast: Pronóstico de régimen del forecaster
            market_data: Datos de mercado (OHLCV)
            features: Features adicionales para filtros
            
        Returns:
            DataFrame con señales de trading
            
        Raises:
            ValueError: Si los datos no son válidos
        """
        logger.info("Generando señales de trading desde pronóstico de régimen")
        
        # Validar entradas
        if not regime_forecast or 'current_regime' not in regime_forecast:
            raise ValueError("Pronóstico de régimen inválido")
        
        if market_data.empty:
            raise ValueError("Datos de mercado vacíos")
        
        # Obtener información de régimen actual
        current_regime_info = regime_forecast['current_regime']
        ensemble_forecast = regime_forecast.get('ensemble_forecast', {})
        
        # Crear DataFrame de señales
        signals_data = []
        
        # Para cada horizonte en el pronóstico
        horizons = ensemble_forecast.get('horizons', {})
        if not horizons:
            # Usar solo régimen actual
            horizons = {1: {
                'state_probabilities': current_regime_info.get('ensemble_probabilities', []),
                'most_likely_state': current_regime_info.get('ensemble_regime', 0)
            }}
        
        for horizon, horizon_data in horizons.items():
            # Extraer probabilidades de estado
            state_probs = horizon_data.get('state_probabilities', [])
            most_likely_state = horizon_data.get('most_likely_state', 0)
            
            if not state_probs:
                logger.warning(f"No hay probabilidades de estado para horizonte {horizon}")
                continue
            
            # Generar señal base
            base_signal = self._map_regime_to_signal(most_likely_state, state_probs)
            
            # Aplicar filtros
            filtered_signal = self._apply_signal_filters(
                base_signal, market_data, features, current_regime_info
            )
            
            # Calcular position size
            position_size = self._calculate_position_size(
                filtered_signal, state_probs, market_data, features
            )
            
            # Calcular métricas de confianza
            signal_confidence = self._calculate_signal_confidence(
                state_probs, current_regime_info
            )
            
            # Crear registro de señal
            signal_record = {
                'timestamp': datetime.now(),
                'horizon': horizon,
                'regime_state': most_likely_state,
                'regime_probabilities': state_probs,
                'base_signal': base_signal['type'].value,
                'base_strength': base_signal['strength'].value,
                'filtered_signal': filtered_signal['type'].value,
                'filtered_strength': filtered_signal['strength'].value,
                'position_size': position_size,
                'signal_confidence': signal_confidence,
                'filters_applied': filtered_signal.get('filters_applied', []),
                'regime_confidence': current_regime_info.get('confidence', 0.0)
            }
            
            signals_data.append(signal_record)
        
        # Convertir a DataFrame
        signals_df = pd.DataFrame(signals_data)
        
        # Guardar en historial
        self.signal_history.extend(signals_data)
        
        # Actualizar últimas señales
        if not signals_df.empty:
            latest_signal = signals_df.iloc[-1]
            self.last_signals = {
                'signal_type': latest_signal['filtered_signal'],
                'position_size': latest_signal['position_size'],
                'confidence': latest_signal['signal_confidence'],
                'timestamp': latest_signal['timestamp']
            }
        
        logger.info(f"Generadas {len(signals_df)} señales para horizontes: {list(horizons.keys())}")
        
        return signals_df
    
    def _map_regime_to_signal(self, regime_state: int, 
                             state_probabilities: List[float]) -> Dict[str, Any]:
        """
        Mapea régimen a señal de trading.
        
        Args:
            regime_state: Estado de régimen más probable
            state_probabilities: Probabilidades de todos los estados
            
        Returns:
            Señal de trading con tipo y fuerza
        """
        # Obtener acción del mapeo de configuración
        action = self.regime_mapping.get(regime_state, 'flat')
        
        # Convertir a SignalType
        if action == 'long':
            signal_type = SignalType.LONG
        elif action == 'short':
            signal_type = SignalType.SHORT
        else:
            signal_type = SignalType.FLAT
        
        # Calcular fuerza basada en probabilidad del régimen
        regime_prob = state_probabilities[regime_state] if regime_state < len(state_probabilities) else 0.0
        
        if regime_prob >= 0.9:
            strength = SignalStrength.VERY_STRONG
        elif regime_prob >= 0.75:
            strength = SignalStrength.STRONG
        elif regime_prob >= 0.6:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK
        
        return {
            'type': signal_type,
            'strength': strength,
            'regime_probability': regime_prob,
            'regime_state': regime_state,
            'mapped_action': action
        }
    
    def _apply_signal_filters(self, base_signal: Dict[str, Any],
                            market_data: pd.DataFrame,
                            features: Optional[pd.DataFrame],
                            regime_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aplica filtros a la señal base.
        
        Args:
            base_signal: Señal base sin filtrar
            market_data: Datos de mercado
            features: Features adicionales
            regime_info: Información del régimen actual
            
        Returns:
            Señal filtrada
        """
        filtered_signal = base_signal.copy()
        filters_applied = []
        
        # Filtro de probabilidad mínima
        if base_signal['regime_probability'] < self.min_regime_prob:
            filtered_signal['type'] = SignalType.FLAT
            filtered_signal['strength'] = SignalStrength.WEAK
            filters_applied.append('min_probability')
            logger.debug(f"Señal filtrada por probabilidad baja: {base_signal['regime_probability']:.3f}")
        
        # Filtro de volatilidad
        if self.vol_filter and features is not None:
            vol_filter_result = self._apply_volatility_filter(
                filtered_signal, market_data, features
            )
            if vol_filter_result['filtered']:
                filtered_signal['type'] = vol_filter_result['new_signal']
                filtered_signal['strength'] = vol_filter_result['new_strength']
                filters_applied.append('volatility')
                logger.debug("Señal filtrada por alta volatilidad")
        
        # Filtro de confianza del régimen
        regime_confidence = regime_info.get('confidence', 1.0)
        if regime_confidence < 0.5:
            # Reducir fuerza de la señal si hay baja confianza
            current_strength = filtered_signal['strength'].value
            new_strength_value = current_strength * 0.5
            
            if new_strength_value < 0.25:
                filtered_signal['type'] = SignalType.FLAT
                filtered_signal['strength'] = SignalStrength.WEAK
            else:
                filtered_signal['strength'] = SignalStrength(min(new_strength_value, 1.0))
            
            filters_applied.append('regime_confidence')
            logger.debug(f"Señal filtrada por baja confianza de régimen: {regime_confidence:.3f}")
        
        # Filtro de momentum (si hay datos suficientes)
        if len(market_data) >= 5:
            momentum_filter_result = self._apply_momentum_filter(
                filtered_signal, market_data
            )
            if momentum_filter_result['filtered']:
                filtered_signal['strength'] = momentum_filter_result['new_strength']
                filters_applied.append('momentum')
        
        filtered_signal['filters_applied'] = filters_applied
        
        return filtered_signal
    
    def _apply_volatility_filter(self, signal: Dict[str, Any],
                               market_data: pd.DataFrame,
                               features: pd.DataFrame) -> Dict[str, Any]:
        """
        Aplica filtro de volatilidad.
        
        Args:
            signal: Señal a filtrar
            market_data: Datos de mercado
            features: Features con volatilidad
            
        Returns:
            Resultado del filtro
        """
        # Buscar columna de volatilidad
        vol_columns = [col for col in features.columns if 'volatility' in col.lower()]
        if not vol_columns:
            return {'filtered': False}
        
        vol_column = vol_columns[0]
        current_vol = features[vol_column].iloc[-1]
        
        # Calcular volatilidad promedio
        vol_window = min(50, len(features))
        avg_vol = features[vol_column].tail(vol_window).mean()
        
        # Verificar si volatilidad está por encima del umbral
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
        
        if vol_ratio > self.vol_threshold:
            # Alta volatilidad - reducir exposición o evitar trading
            return {
                'filtered': True,
                'new_signal': SignalType.FLAT,
                'new_strength': SignalStrength.WEAK,
                'vol_ratio': vol_ratio
            }
        
        return {'filtered': False, 'vol_ratio': vol_ratio}
    
    def _apply_momentum_filter(self, signal: Dict[str, Any],
                             market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Aplica filtro de momentum.
        
        Args:
            signal: Señal a filtrar
            market_data: Datos de mercado
            
        Returns:
            Resultado del filtro
        """
        # Calcular momentum simple (retorno de 5 períodos)
        if len(market_data) < 5:
            return {'filtered': False}
        
        recent_prices = market_data['close'].tail(5)
        momentum = (recent_prices.iloc[-1] / recent_prices.iloc[0]) - 1
        
        # Ajustar fuerza basada en momentum
        if signal['type'] == SignalType.LONG and momentum < -0.05:
            # Long signal pero momentum negativo
            new_strength_value = signal['strength'].value * 0.5
            new_strength = SignalStrength(max(new_strength_value, 0.25))
            return {
                'filtered': True,
                'new_strength': new_strength,
                'momentum': momentum
            }
        elif signal['type'] == SignalType.SHORT and momentum > 0.05:
            # Short signal pero momentum positivo
            new_strength_value = signal['strength'].value * 0.5
            new_strength = SignalStrength(max(new_strength_value, 0.25))
            return {
                'filtered': True,
                'new_strength': new_strength,
                'momentum': momentum
            }
        
        return {'filtered': False, 'momentum': momentum}
    
    def _calculate_position_size(self, signal: Dict[str, Any],
                               state_probabilities: List[float],
                               market_data: pd.DataFrame,
                               features: Optional[pd.DataFrame]) -> float:
        """
        Calcula tamaño de posición.
        
        Args:
            signal: Señal de trading
            state_probabilities: Probabilidades de estado
            market_data: Datos de mercado
            features: Features adicionales
            
        Returns:
            Tamaño de posición (fracción del capital)
        """
        if signal['type'] == SignalType.FLAT:
            return 0.0
        
        base_size = 0.0
        
        if self.sizing_method == 'fixed':
            # Tamaño fijo basado en fuerza de señal
            base_size = signal['strength'].value * 0.1
            
        elif self.sizing_method == 'kelly_fraction':
            # Kelly fraction simplificado
            kelly_size = self._calculate_kelly_fraction(
                signal, state_probabilities, market_data, features
            )
            base_size = min(kelly_size, self.kelly_fraction_limit)
            
        elif self.sizing_method == 'vol_target':
            # Volatility targeting
            vol_target_size = self._calculate_vol_target_size(
                signal, market_data, features
            )
            base_size = vol_target_size
        
        # Ajustar por fuerza de señal
        adjusted_size = base_size * signal['strength'].value
        
        # Aplicar límites
        final_size = np.clip(adjusted_size, 0.0, self.max_leverage)
        
        # Considerar dirección
        if signal['type'] == SignalType.SHORT:
            final_size = -final_size
        
        return final_size
    
    def _calculate_kelly_fraction(self, signal: Dict[str, Any],
                                state_probabilities: List[float],
                                market_data: pd.DataFrame,
                                features: Optional[pd.DataFrame]) -> float:
        """
        Calcula Kelly fraction basado en probabilidades de régimen.
        
        Args:
            signal: Señal de trading
            state_probabilities: Probabilidades de estado
            market_data: Datos de mercado
            features: Features adicionales
            
        Returns:
            Kelly fraction
        """
        # Estimar probabilidad de éxito y ratio win/loss
        # Esto es una aproximación - en implementación real se usarían
        # estadísticas históricas por régimen
        
        regime_prob = signal['regime_probability']
        
        # Probabilidad de éxito aproximada basada en confianza del régimen
        win_prob = 0.5 + (regime_prob - 0.5) * 0.3  # Ajuste conservador
        
        # Ratio win/loss aproximado (se estimaría de datos históricos)
        win_loss_ratio = 1.2  # Ligeramente positivo
        
        # Kelly fraction = (b*p - q) / b
        # donde b = win/loss ratio, p = win prob, q = loss prob
        loss_prob = 1 - win_prob
        kelly = (win_loss_ratio * win_prob - loss_prob) / win_loss_ratio
        
        # Aplicar factor de reducción para Kelly fraccional
        kelly_fractional = kelly * 0.25  # Factor conservador
        
        return max(0.0, kelly_fractional)
    
    def _calculate_vol_target_size(self, signal: Dict[str, Any],
                                 market_data: pd.DataFrame,
                                 features: Optional[pd.DataFrame]) -> float:
        """
        Calcula tamaño basado en volatility targeting.
        
        Args:
            signal: Señal de trading
            market_data: Datos de mercado
            features: Features adicionales
            
        Returns:
            Tamaño de posición
        """
        # Estimar volatilidad actual
        if features is not None:
            vol_columns = [col for col in features.columns if 'volatility' in col.lower()]
            if vol_columns:
                current_vol = features[vol_columns[0]].iloc[-1]
            else:
                # Fallback a volatilidad de retornos
                returns = market_data['close'].pct_change().dropna()
                current_vol = returns.tail(20).std() * np.sqrt(252)  # Anualizada
        else:
            returns = market_data['close'].pct_change().dropna()
            current_vol = returns.tail(20).std() * np.sqrt(252)
        
        if current_vol <= 0:
            return 0.0
        
        # Position size = target_vol / current_vol
        position_size = self.vol_target / current_vol
        
        return min(position_size, 1.0)  # Máximo 100% del capital
    
    def _calculate_signal_confidence(self, state_probabilities: List[float],
                                   regime_info: Dict[str, Any]) -> float:
        """
        Calcula confianza en la señal.
        
        Args:
            state_probabilities: Probabilidades de estado
            regime_info: Información del régimen
            
        Returns:
            Nivel de confianza [0, 1]
        """
        # Confianza basada en máxima probabilidad
        max_prob = max(state_probabilities) if state_probabilities else 0.0
        
        # Confianza del régimen
        regime_confidence = regime_info.get('confidence', 0.0)
        
        # Diversidad de modelos (si está disponible)
        n_models = regime_info.get('n_models_consensus', 1)
        model_diversity_factor = min(n_models / 3.0, 1.0)  # Máximo factor por 3+ modelos
        
        # Combinar factores
        combined_confidence = (
            max_prob * 0.5 +
            regime_confidence * 0.3 +
            model_diversity_factor * 0.2
        )
        
        return np.clip(combined_confidence, 0.0, 1.0)
    
    def get_current_signal(self) -> Optional[Dict[str, Any]]:
        """
        Obtiene la señal actual.
        
        Returns:
            Señal actual o None si no hay
        """
        return self.last_signals.copy() if self.last_signals else None
    
    def get_signal_history(self, n_last: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Obtiene historial de señales.
        
        Args:
            n_last: Número de últimas señales (todas si None)
            
        Returns:
            Historial de señales
        """
        if n_last is None:
            return self.signal_history.copy()
        else:
            return self.signal_history[-n_last:].copy()
    
    def clear_signal_history(self):
        """Limpia el historial de señales."""
        self.signal_history.clear()
        self.last_signals.clear()
        logger.info("Historial de señales limpiado")
    
    def update_regime_mapping(self, new_mapping: Dict[int, str]):
        """
        Actualiza el mapeo de régimen a señal.
        
        Args:
            new_mapping: Nuevo mapeo {regime_id: action}
        """
        valid_actions = ['long', 'short', 'flat']
        
        for regime, action in new_mapping.items():
            if action not in valid_actions:
                raise ValueError(f"Acción '{action}' no válida. Válidas: {valid_actions}")
        
        self.regime_mapping = new_mapping.copy()
        logger.info(f"Mapeo de régimen actualizado: {self.regime_mapping}")
    
    def get_position_sizing_info(self) -> Dict[str, Any]:
        """
        Obtiene información de configuración de position sizing.
        
        Returns:
            Información de position sizing
        """
        return {
            'method': self.sizing_method,
            'kelly_fraction_limit': self.kelly_fraction_limit,
            'vol_target': self.vol_target,
            'max_leverage': self.max_leverage,
            'min_regime_prob': self.min_regime_prob,
            'vol_filter': self.vol_filter,
            'vol_threshold': self.vol_threshold
        }


if __name__ == "__main__":
    # Ejemplo de uso
    try:
        from ..config_loader import load_config
        
        config = load_config("config/binance_spot_example.yaml")
        
        # Crear lógica de posiciones
        position_logic = PositionLogic(config)
        
        # Simular pronóstico de régimen
        regime_forecast = {
            'current_regime': {
                'ensemble_regime': 1,
                'ensemble_probabilities': [0.2, 0.7, 0.1],
                'confidence': 0.8,
                'n_models_consensus': 2
            },
            'ensemble_forecast': {
                'horizons': {
                    1: {
                        'state_probabilities': [0.2, 0.7, 0.1],
                        'most_likely_state': 1
                    },
                    3: {
                        'state_probabilities': [0.3, 0.5, 0.2],
                        'most_likely_state': 1
                    }
                }
            }
        }
        
        # Crear datos de mercado simulados
        np.random.seed(42)
        n_samples = 50
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='1H')
        
        market_data = pd.DataFrame({
            'open': np.random.normal(100, 5, n_samples),
            'high': np.random.normal(102, 5, n_samples),
            'low': np.random.normal(98, 5, n_samples),
            'close': np.random.normal(100, 5, n_samples),
            'volume': np.random.lognormal(10, 1, n_samples)
        }, index=dates)
        
        # Crear features simuladas
        features = pd.DataFrame({
            'log_return_1': np.random.normal(0, 0.02, n_samples),
            'volatility': np.random.normal(0.02, 0.005, n_samples),
            'volume_zscore': np.random.normal(0, 1, n_samples)
        }, index=dates)
        
        print("Configuración de position logic:")
        print(f"  Mapeo de régimen: {position_logic.regime_mapping}")
        print(f"  Método de sizing: {position_logic.sizing_method}")
        
        # Generar señales
        signals = position_logic.generate_signals(regime_forecast, market_data, features)
        
        print(f"\nSeñales generadas: {len(signals)}")
        for idx, signal in signals.iterrows():
            print(f"  Horizonte {signal['horizon']}: "
                  f"Señal={signal['filtered_signal']}, "
                  f"Size={signal['position_size']:.3f}, "
                  f"Confianza={signal['signal_confidence']:.3f}")
        
        # Mostrar información de la señal actual
        current = position_logic.get_current_signal()
        if current:
            print(f"\nSeñal actual:")
            print(f"  Tipo: {current['signal_type']}")
            print(f"  Tamaño: {current['position_size']:.3f}")
            print(f"  Confianza: {current['confidence']:.3f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
