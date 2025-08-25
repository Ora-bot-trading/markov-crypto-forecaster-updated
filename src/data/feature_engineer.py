"""
Ingeniería de features para markov-crypto-forecaster.

Crea features técnicas sin look-ahead bias para modelos de Markov.
Incluye retornos log, volatilidad, ATR, order flow imbalance y z-scores.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Union, List, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
import warnings
from numba import jit, njit
import talib

from ..config_loader import Config
from ..logging_utils import get_logger, log_execution_time
from ..utils_time import timeframe_to_minutes

logger = get_logger(__name__)


@njit
def rolling_z_score_numba(values: np.ndarray, window: int) -> np.ndarray:
    """
    Calcula z-score rolling optimizado con numba.
    
    Args:
        values: Array de valores
        window: Ventana para cálculo
        
    Returns:
        Array con z-scores
    """
    n = len(values)
    result = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        window_data = values[i - window + 1:i + 1]
        mean_val = np.mean(window_data)
        std_val = np.std(window_data)
        
        if std_val > 0:
            result[i] = (values[i] - mean_val) / std_val
        else:
            result[i] = 0.0
    
    return result


@njit
def realized_volatility_numba(log_returns: np.ndarray, window: int) -> np.ndarray:
    """
    Calcula volatilidad realizada con numba.
    
    Args:
        log_returns: Array de retornos log
        window: Ventana para cálculo
        
    Returns:
        Array con volatilidad realizada
    """
    n = len(log_returns)
    result = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        window_returns = log_returns[i - window + 1:i + 1]
        # Filtrar NaN
        valid_returns = window_returns[~np.isnan(window_returns)]
        if len(valid_returns) > 1:
            result[i] = np.sqrt(np.sum(valid_returns ** 2))
        
    return result


class FeatureEngineer:
    """
    Ingeniería de features para trading de criptomonedas.
    
    Crea features técnicas sin look-ahead bias, incluyendo:
    - Retornos log multi-período
    - Volatilidades rolling
    - Indicadores técnicos (ATR, RSI)
    - Z-scores rolling
    - Order flow features
    - Features de microestructura
    """
    
    def __init__(self, config: Union[Config, dict]):
        """
        Inicializa el ingeniero de features.
        
        Args:
            config: Configuración del sistema
        """
        if isinstance(config, dict):
            self.config = config
        else:
            self.config = config.dict()
            
        self.features_config = self.config.get('features', {})
        self.paths_config = self.config.get('paths', {})
        
        # Configuración de features
        self.enabled_features = self.features_config.get('enabled', [])
        self.feature_params = self.features_config.get('params', {})
        self.discretization_config = self.features_config.get('discretization', {})
        
        # Parámetros por defecto
        self.volatility_window = self.feature_params.get('volatility_window', 20)
        self.atr_window = self.feature_params.get('atr_window', 14)
        self.rv_window = self.feature_params.get('rv_window', 20)
        self.zscore_window = self.feature_params.get('zscore_window', 50)
        
        # Crear directorios
        self.features_data_dir = Path(self.paths_config.get('data_dir', 'data')) / 'features'
        self.features_data_dir.mkdir(parents=True, exist_ok=True)
        
        self.scalers = {}
        self.feature_metadata = {}
    
    @log_execution_time("create_log_returns")
    def create_log_returns(self, df: pd.DataFrame, 
                          periods: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """
        Crea retornos logarítmicos para múltiples períodos.
        
        Args:
            df: DataFrame con datos OHLCV
            periods: Períodos para calcular retornos
            
        Returns:
            DataFrame con features de retornos
        """
        features = pd.DataFrame(index=df.index)
        
        # Usar precio close para retornos
        close_prices = df['close']
        
        for period in periods:
            # Retorno log
            log_ret = np.log(close_prices / close_prices.shift(period))
            features[f'log_return_{period}'] = log_ret
            
            # Retorno simple para comparación
            simple_ret = (close_prices / close_prices.shift(period)) - 1
            features[f'simple_return_{period}'] = simple_ret
            
            # Retorno absoluto (proxy de volatilidad)
            features[f'abs_return_{period}'] = np.abs(log_ret)
        
        # Retorno forward para targets (cuidado: NO usar en features de entrada)
        features['forward_return_1'] = np.log(close_prices.shift(-1) / close_prices)
        
        logger.info(f"Creados retornos log para períodos: {periods}")
        
        return features
    
    @log_execution_time("create_volatility_features")
    def create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features de volatilidad.
        
        Args:
            df: DataFrame con datos OHLCV
            
        Returns:
            DataFrame con features de volatilidad
        """
        features = pd.DataFrame(index=df.index)
        
        # Retornos log para cálculo de volatilidad
        log_returns = np.log(df['close'] / df['close'].shift(1))
        
        # 1. Volatilidad rolling estándar
        vol_std = log_returns.rolling(window=self.volatility_window, min_periods=5).std()
        features['volatility_std'] = vol_std
        
        # 2. Volatilidad anualizada (asumir trading 24/7 para crypto)
        periods_per_year = 365 * 24 * (60 / timeframe_to_minutes(self.config.get('data', {}).get('timeframe', '1h')))
        features['volatility_annualized'] = vol_std * np.sqrt(periods_per_year)
        
        # 3. Volatilidad realizada usando numba
        rv = realized_volatility_numba(log_returns.fillna(0).values, self.rv_window)
        features['realized_volatility'] = rv
        
        # 4. Garman-Klass volatility estimator (más eficiente)
        if all(col in df.columns for col in ['high', 'low', 'open', 'close']):
            gk_vol = self._garman_klass_volatility(df)
            features['gk_volatility'] = gk_vol
        
        # 5. Parkinson volatility estimator
        if all(col in df.columns for col in ['high', 'low']):
            park_vol = self._parkinson_volatility(df)
            features['parkinson_volatility'] = park_vol
        
        # 6. Volatilidad de volumen
        volume_log_ret = np.log(df['volume'] / df['volume'].shift(1))
        vol_volume = volume_log_ret.rolling(window=self.volatility_window, min_periods=5).std()
        features['volume_volatility'] = vol_volume
        
        logger.info(f"Creadas {len([col for col in features.columns if 'volatility' in col])} features de volatilidad")
        
        return features
    
    def _garman_klass_volatility(self, df: pd.DataFrame) -> pd.Series:
        """
        Calcula volatilidad Garman-Klass.
        
        Args:
            df: DataFrame con OHLC
            
        Returns:
            Series con volatilidad GK
        """
        hl = np.log(df['high'] / df['low'])
        co = np.log(df['close'] / df['open'])
        
        gk_vol = 0.5 * hl**2 - (2*np.log(2) - 1) * co**2
        
        return gk_vol.rolling(window=self.volatility_window, min_periods=5).mean().apply(np.sqrt)
    
    def _parkinson_volatility(self, df: pd.DataFrame) -> pd.Series:
        """
        Calcula volatilidad Parkinson.
        
        Args:
            df: DataFrame con HL
            
        Returns:
            Series con volatilidad Parkinson
        """
        hl_ratio = np.log(df['high'] / df['low'])
        park_vol = hl_ratio**2 / (4 * np.log(2))
        
        return park_vol.rolling(window=self.volatility_window, min_periods=5).mean().apply(np.sqrt)
    
    @log_execution_time("create_technical_indicators")
    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea indicadores técnicos tradicionales.
        
        Args:
            df: DataFrame con datos OHLCV
            
        Returns:
            DataFrame con indicadores técnicos
        """
        features = pd.DataFrame(index=df.index)
        
        # Convertir a numpy arrays para TA-Lib
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values
        
        try:
            # 1. Average True Range (ATR)
            atr = talib.ATR(high, low, close, timeperiod=self.atr_window)
            features['atr'] = atr
            features['atr_pct'] = atr / close  # ATR como % del precio
            
            # 2. RSI (solo informativo, no para señales directas)
            rsi = talib.RSI(close, timeperiod=14)
            features['rsi'] = rsi
            features['rsi_normalized'] = (rsi - 50) / 50  # Normalizado [-1, 1]
            
            # 3. Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            features['bb_upper'] = bb_upper
            features['bb_lower'] = bb_lower
            features['bb_width'] = (bb_upper - bb_lower) / bb_middle
            features['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)
            
            # 4. MACD
            macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            features['macd'] = macd
            features['macd_signal'] = macd_signal
            features['macd_histogram'] = macd_hist
            
            # 5. Stochastic
            stoch_k, stoch_d = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
            features['stoch_k'] = stoch_k
            features['stoch_d'] = stoch_d
            
            # 6. Williams %R
            williams_r = talib.WILLR(high, low, close, timeperiod=14)
            features['williams_r'] = williams_r
            
            # 7. Commodity Channel Index
            cci = talib.CCI(high, low, close, timeperiod=14)
            features['cci'] = cci
            
        except Exception as e:
            logger.warning(f"Error calculando indicadores técnicos: {e}")
            # Crear versiones simplificadas sin TA-Lib
            features['atr'] = self._simple_atr(df)
            features['rsi'] = self._simple_rsi(df)
        
        logger.info(f"Creados {len(features.columns)} indicadores técnicos")
        
        return features
    
    def _simple_atr(self, df: pd.DataFrame) -> pd.Series:
        """ATR simple sin TA-Lib."""
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift(1)).abs()
        tr3 = (df['low'] - df['close'].shift(1)).abs()
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=self.atr_window, min_periods=1).mean()
    
    def _simple_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """RSI simple sin TA-Lib."""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @log_execution_time("create_volume_features")
    def create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features basadas en volumen.
        
        Args:
            df: DataFrame con datos OHLCV
            
        Returns:
            DataFrame con features de volumen
        """
        features = pd.DataFrame(index=df.index)
        
        volume = df['volume']
        close = df['close']
        
        # 1. Volume features básicas
        features['volume'] = volume
        features['volume_log'] = np.log(volume + 1)  # +1 para evitar log(0)
        
        # 2. Volume moving averages
        features['volume_ma_short'] = volume.rolling(window=10).mean()
        features['volume_ma_long'] = volume.rolling(window=30).mean()
        features['volume_ratio'] = features['volume_ma_short'] / features['volume_ma_long']
        
        # 3. Volume Rate of Change
        features['volume_roc'] = volume.pct_change(periods=10)
        
        # 4. Volume-Price Trend (VPT)
        price_change_pct = close.pct_change()
        vpt = (price_change_pct * volume).cumsum()
        features['vpt'] = vpt
        
        # 5. Volume delta (aproximación sin order book)
        # Usar precio para inferir dirección del volumen
        price_direction = np.where(close > close.shift(1), 1, 
                          np.where(close < close.shift(1), -1, 0))
        volume_delta = volume * price_direction
        features['volume_delta'] = volume_delta
        features['volume_delta_ma'] = volume_delta.rolling(window=20).mean()
        
        # 6. Volume Z-score
        vol_zscore = rolling_z_score_numba(volume.fillna(volume.median()).values, self.zscore_window)
        features['volume_zscore'] = vol_zscore
        
        # 7. Volume percentile
        features['volume_percentile'] = volume.rolling(window=100).rank(pct=True)
        
        logger.info(f"Creadas {len(features.columns)} features de volumen")
        
        return features
    
    @log_execution_time("create_price_features")
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features adicionales de precio.
        
        Args:
            df: DataFrame con datos OHLCV
            
        Returns:
            DataFrame con features de precio
        """
        features = pd.DataFrame(index=df.index)
        
        # 1. Price ranges
        features['high_low_ratio'] = df['high'] / df['low']
        features['open_close_ratio'] = df['open'] / df['close']
        features['hl_pct'] = (df['high'] - df['low']) / df['close']
        
        # 2. Gap features
        features['gap_up'] = np.maximum(0, df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        features['gap_down'] = np.maximum(0, df['close'].shift(1) - df['open']) / df['close'].shift(1)
        
        # 3. Doji patterns (approximation)
        body_size = np.abs(df['close'] - df['open'])
        full_range = df['high'] - df['low']
        features['doji_ratio'] = body_size / (full_range + 1e-8)  # Small value to avoid division by zero
        
        # 4. Upper/Lower shadows
        upper_shadow = df['high'] - np.maximum(df['open'], df['close'])
        lower_shadow = np.minimum(df['open'], df['close']) - df['low']
        features['upper_shadow_pct'] = upper_shadow / full_range
        features['lower_shadow_pct'] = lower_shadow / full_range
        
        # 5. Price position dentro del rango
        features['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        logger.info(f"Creadas {len(features.columns)} features de precio")
        
        return features
    
    @log_execution_time("create_z_scores")
    def create_z_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea z-scores rolling para detección de anomalías.
        
        Args:
            df: DataFrame con datos OHLCV
            
        Returns:
            DataFrame con z-scores
        """
        features = pd.DataFrame(index=df.index)
        
        # Variables clave para z-scores
        key_vars = {
            'close': df['close'],
            'volume': df['volume'],
            'high_low_ratio': df['high'] / df['low']
        }
        
        # Agregar retorno si no existe
        if 'log_return_1' not in df.columns:
            key_vars['log_return_1'] = np.log(df['close'] / df['close'].shift(1))
        else:
            key_vars['log_return_1'] = df['log_return_1']
        
        for var_name, var_series in key_vars.items():
            if var_series.isna().all():
                continue
                
            # Z-score usando rolling mean y std
            zscore = rolling_z_score_numba(var_series.fillna(var_series.median()).values, self.zscore_window)
            features[f'{var_name}_zscore'] = zscore
            
            # Z-score absoluto (magnitud de anomalía)
            features[f'{var_name}_zscore_abs'] = np.abs(zscore)
            
            # Percentile rolling
            percentile = var_series.rolling(window=self.zscore_window).rank(pct=True)
            features[f'{var_name}_percentile'] = percentile
        
        logger.info(f"Creados z-scores para {len(key_vars)} variables")
        
        return features
    
    def create_order_flow_features(self, df: pd.DataFrame, 
                                 level2_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Crea features de order flow (requiere datos de profundidad nivel 2).
        
        Args:
            df: DataFrame con datos OHLCV
            level2_data: Datos de order book (opcional)
            
        Returns:
            DataFrame con features de order flow
        """
        features = pd.DataFrame(index=df.index)
        
        if level2_data is not None and not level2_data.empty:
            # Order flow imbalance real
            logger.info("Calculando order flow imbalance con datos L2")
            # TODO: Implementar cálculo real con order book
            # Por ahora, aproximación usando price/volume
        
        # Aproximaciones sin datos L2
        close = df['close']
        volume = df['volume']
        
        # 1. Volume-weighted average price (VWAP) aproximado
        # Usar typical price como proxy
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * volume).rolling(window=20).sum() / volume.rolling(window=20).sum()
        features['vwap'] = vwap
        features['price_vs_vwap'] = close / vwap - 1
        
        # 2. Money flow
        money_flow = typical_price * volume
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        features['money_flow_ratio'] = (
            positive_flow.rolling(window=14).sum() / 
            (negative_flow.rolling(window=14).sum() + 1e-8)
        )
        
        # 3. Order flow imbalance proxy
        # Usar volume y direction del precio
        price_direction = np.sign(close.diff())
        ofi_proxy = (volume * price_direction).rolling(window=10).sum()
        features['ofi_proxy'] = ofi_proxy
        
        logger.info(f"Creadas {len(features.columns)} features de order flow (aproximadas)")
        
        return features
    
    def discretize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Discretiza features para HMM discreto.
        
        Args:
            df: DataFrame con features continuas
            
        Returns:
            DataFrame con features discretizadas
        """
        if not self.discretization_config:
            logger.info("No hay configuración de discretización")
            return pd.DataFrame(index=df.index)
        
        n_quantiles = self.discretization_config.get('n_quantiles', 5)
        features_to_discretize = self.discretization_config.get('features_to_discretize', [])
        
        discretized = pd.DataFrame(index=df.index)
        
        for feature in features_to_discretize:
            if feature not in df.columns:
                logger.warning(f"Feature '{feature}' no encontrada para discretización")
                continue
            
            # Usar QuantileTransformer para discretización robusta
            series = df[feature].dropna()
            if len(series) < 50:  # Necesitamos datos suficientes
                logger.warning(f"Pocos datos para discretizar '{feature}': {len(series)}")
                continue
            
            transformer = QuantileTransformer(
                n_quantiles=min(n_quantiles, len(series)),
                output_distribution='uniform',
                random_state=42
            )
            
            # Fit solo en datos disponibles
            transformed = transformer.fit_transform(series.values.reshape(-1, 1)).flatten()
            
            # Convertir a bins discretos
            bins = np.linspace(0, 1, n_quantiles + 1)
            discrete_values = np.digitize(transformed, bins) - 1  # 0-indexed
            discrete_values = np.clip(discrete_values, 0, n_quantiles - 1)
            
            # Crear serie completa con NaN donde corresponda
            discrete_series = pd.Series(index=df.index, dtype=float)
            discrete_series.loc[series.index] = discrete_values
            
            discretized[f'{feature}_discrete'] = discrete_series
        
        logger.info(f"Discretizadas {len(discretized.columns)} features")
        
        return discretized
    
    @log_execution_time("create_all_features")
    def create_all_features(self, df: pd.DataFrame,
                           include_target: bool = True,
                           level2_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Crea todas las features habilitadas.
        
        Args:
            df: DataFrame con datos OHLCV
            include_target: Si incluir variables target (forward returns)
            level2_data: Datos de order book nivel 2 (opcional)
            
        Returns:
            DataFrame con todas las features
        """
        logger.info(f"Creando features habilitadas: {self.enabled_features}")
        
        all_features = pd.DataFrame(index=df.index)
        
        # Crear features según configuración
        if 'log_returns' in self.enabled_features:
            returns_features = self.create_log_returns(df)
            all_features = pd.concat([all_features, returns_features], axis=1)
        
        if 'volatility' in self.enabled_features:
            vol_features = self.create_volatility_features(df)
            all_features = pd.concat([all_features, vol_features], axis=1)
        
        if 'atr' in self.enabled_features or 'rsi' in self.enabled_features:
            tech_features = self.create_technical_indicators(df)
            all_features = pd.concat([all_features, tech_features], axis=1)
        
        if 'volume_delta' in self.enabled_features:
            volume_features = self.create_volume_features(df)
            all_features = pd.concat([all_features, volume_features], axis=1)
        
        if 'realized_vol' in self.enabled_features:
            price_features = self.create_price_features(df)
            all_features = pd.concat([all_features, price_features], axis=1)
        
        if 'z_scores' in self.enabled_features:
            zscore_features = self.create_z_scores(all_features if len(all_features.columns) > 0 else df)
            all_features = pd.concat([all_features, zscore_features], axis=1)
        
        if 'order_flow_imbalance' in self.enabled_features:
            ofi_features = self.create_order_flow_features(df, level2_data)
            all_features = pd.concat([all_features, ofi_features], axis=1)
        
        # Crear features discretizadas si se requieren
        if self.discretization_config:
            discrete_features = self.discretize_features(all_features)
            all_features = pd.concat([all_features, discrete_features], axis=1)
        
        # Remover columnas con todos NaN
        all_features = all_features.dropna(axis=1, how='all')
        
        # Remover features target si no se requieren
        if not include_target:
            target_columns = [col for col in all_features.columns if 'forward_return' in col]
            all_features = all_features.drop(columns=target_columns, errors='ignore')
        
        logger.info(f"Creadas {len(all_features.columns)} features totales")
        
        # Guardar metadata de features
        self.feature_metadata = {
            'total_features': len(all_features.columns),
            'feature_types': {
                'returns': len([col for col in all_features.columns if 'return' in col]),
                'volatility': len([col for col in all_features.columns if 'volatility' in col or 'vol' in col]),
                'technical': len([col for col in all_features.columns if any(tech in col for tech in ['atr', 'rsi', 'macd', 'bb_'])]),
                'volume': len([col for col in all_features.columns if 'volume' in col]),
                'zscore': len([col for col in all_features.columns if 'zscore' in col]),
                'discrete': len([col for col in all_features.columns if 'discrete' in col])
            },
            'enabled_features': self.enabled_features,
            'feature_params': self.feature_params
        }
        
        return all_features
    
    def save_features(self, features_df: pd.DataFrame,
                     symbol: str,
                     timeframe: str,
                     suffix: str = "") -> Path:
        """
        Guarda features calculadas.
        
        Args:
            features_df: DataFrame con features
            symbol: Símbolo de los datos
            timeframe: Timeframe de los datos
            suffix: Sufijo para el nombre de archivo
            
        Returns:
            Path del archivo guardado
        """
        safe_symbol = symbol.replace('/', '_')
        
        base_name = f"{safe_symbol}_{timeframe}_features"
        if suffix:
            base_name += f"_{suffix}"
        
        features_path = self.features_data_dir / f"{base_name}.parquet"
        features_df.to_parquet(features_path)
        
        # Guardar metadata
        metadata_path = self.features_data_dir / f"{base_name}_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(self.feature_metadata, f, indent=2)
        
        logger.info(f"Features guardadas: {features_path}")
        
        return features_path


if __name__ == "__main__":
    # Ejemplo de uso
    from ..config_loader import load_config
    
    try:
        config = load_config("config/binance_spot_example.yaml")
        engineer = FeatureEngineer(config)
        
        # Crear datos de ejemplo
        dates = pd.date_range('2023-01-01', periods=1000, freq='1H', tz='UTC')
        np.random.seed(42)
        
        # Simular datos OHLCV
        returns = np.random.normal(0, 0.02, len(dates))
        prices = 100 * np.exp(returns.cumsum())
        
        df = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, len(dates))
        }, index=dates)
        
        print(f"Datos de ejemplo: {len(df)} registros")
        
        # Crear features
        features = engineer.create_all_features(df)
        
        print(f"\nFeatures creadas: {len(features.columns)}")
        print("\nTipos de features:")
        for feature_type, count in engineer.feature_metadata['feature_types'].items():
            print(f"  {feature_type}: {count}")
        
        print(f"\nPrimeras 5 features:")
        print(features.head()[features.columns[:5]])
        
        # Verificar NaN
        nan_counts = features.isna().sum()
        features_with_nans = nan_counts[nan_counts > 0]
        if len(features_with_nans) > 0:
            print(f"\nFeatures con NaN:")
            print(features_with_nans.head(10))
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
