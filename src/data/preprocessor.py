"""
Preprocesador de datos para markov-crypto-forecaster.

Limpieza, validación y preprocesamiento de datos OHLCV sin look-ahead bias.
Incluye detección de outliers, relleno de gaps y splits temporales.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Union, Tuple, List
from scipy import stats
import warnings

from ..config_loader import Config
from ..logging_utils import get_logger, log_execution_time
from ..utils_time import (
    time_series_split_no_leakage, 
    walk_forward_splits,
    validate_data_continuity,
    timeframe_to_minutes,
    align_dataframes
)

logger = get_logger(__name__)


class DataPreprocessor:
    """
    Preprocesador de datos financieros sin look-ahead bias.
    
    Maneja limpieza, validación, relleno de gaps y splits temporales
    manteniendo la integridad temporal de los datos.
    """
    
    def __init__(self, config: Union[Config, dict]):
        """
        Inicializa el preprocesador.
        
        Args:
            config: Configuración del sistema
        """
        if isinstance(config, dict):
            self.config = config
        else:
            self.config = config.dict()
            
        self.data_config = self.config.get('data', {})
        self.preprocessing_config = self.data_config.get('preprocessing', {})
        self.paths_config = self.config.get('paths', {})
        
        # Parámetros de preprocesamiento
        self.remove_gaps = self.preprocessing_config.get('remove_gaps', True)
        self.forward_fill_limit = self.preprocessing_config.get('forward_fill_limit', 3)
        self.remove_outliers = self.preprocessing_config.get('remove_outliers', True)
        self.outlier_threshold = self.preprocessing_config.get('outlier_threshold', 5.0)
        
        # Crear directorios
        self.processed_data_dir = Path(self.paths_config.get('data_dir', 'data')) / 'processed'
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        self.validation_results = {}
    
    @log_execution_time("clean_ohlcv_data")
    def clean_ohlcv_data(self, df: pd.DataFrame, 
                        symbol: str = "unknown",
                        timeframe: str = "1h") -> pd.DataFrame:
        """
        Limpia datos OHLCV con validaciones estrictas.
        
        Args:
            df: DataFrame con datos OHLCV
            symbol: Símbolo para logging
            timeframe: Timeframe para validaciones
            
        Returns:
            DataFrame limpio
            
        Raises:
            ValueError: Si los datos no son válidos
        """
        if df.empty:
            logger.warning(f"DataFrame vacío para {symbol}")
            return df
        
        logger.info(f"Limpiando datos OHLCV de {symbol} {timeframe}: {len(df)} registros")
        
        # Crear copia para no modificar original
        df_clean = df.copy()
        original_length = len(df_clean)
        
        # 1. Validar estructura básica
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df_clean.columns]
        if missing_columns:
            raise ValueError(f"Columnas faltantes: {missing_columns}")
        
        # 2. Verificar tipos de datos
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(df_clean[col]):
                logger.warning(f"Convirtiendo columna {col} a numérico")
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # 3. Validar índice temporal
        if not isinstance(df_clean.index, pd.DatetimeIndex):
            raise ValueError("DataFrame debe tener índice DatetimeIndex")
        
        # 4. Remover registros con valores nulos en columnas críticas
        critical_columns = ['open', 'high', 'low', 'close']
        null_mask = df_clean[critical_columns].isnull().any(axis=1)
        if null_mask.any():
            null_count = null_mask.sum()
            logger.warning(f"Removiendo {null_count} registros con valores nulos")
            df_clean = df_clean[~null_mask]
        
        # 5. Validar relaciones OHLC
        invalid_ohlc = self._validate_ohlc_relationships(df_clean)
        if invalid_ohlc.any():
            invalid_count = invalid_ohlc.sum()
            logger.warning(f"Removiendo {invalid_count} registros con OHLC inválido")
            df_clean = df_clean[~invalid_ohlc]
        
        # 6. Validar precios positivos
        price_columns = ['open', 'high', 'low', 'close']
        negative_prices = (df_clean[price_columns] <= 0).any(axis=1)
        if negative_prices.any():
            negative_count = negative_prices.sum()
            logger.warning(f"Removiendo {negative_count} registros con precios no positivos")
            df_clean = df_clean[~negative_prices]
        
        # 7. Validar volumen no negativo
        negative_volume = df_clean['volume'] < 0
        if negative_volume.any():
            negative_vol_count = negative_volume.sum()
            logger.warning(f"Removiendo {negative_vol_count} registros con volumen negativo")
            df_clean = df_clean[~negative_volume]
        
        # 8. Detectar y manejar outliers
        if self.remove_outliers:
            outlier_mask = self._detect_price_outliers(df_clean, threshold=self.outlier_threshold)
            if outlier_mask.any():
                outlier_count = outlier_mask.sum()
                logger.warning(f"Removiendo {outlier_count} outliers de precio")
                df_clean = df_clean[~outlier_mask]
        
        # 9. Remover duplicados temporales
        duplicate_index = df_clean.index.duplicated(keep='last')
        if duplicate_index.any():
            dup_count = duplicate_index.sum()
            logger.warning(f"Removiendo {dup_count} registros duplicados")
            df_clean = df_clean[~duplicate_index]
        
        # 10. Ordenar por índice temporal
        df_clean = df_clean.sort_index()
        
        # 11. Validar continuidad temporal
        if len(df_clean) > 1:
            continuity_stats = validate_data_continuity(df_clean, timeframe)
            self.validation_results[f"{symbol}_{timeframe}"] = continuity_stats
            
            if continuity_stats['continuity_score'] < 0.95:
                logger.warning(f"Baja continuidad temporal: {continuity_stats['continuity_score']:.2%}")
        
        final_length = len(df_clean)
        removed_count = original_length - final_length
        removed_pct = (removed_count / original_length) * 100 if original_length > 0 else 0
        
        logger.info(f"Limpieza completada: {removed_count} registros removidos ({removed_pct:.1f}%)")
        
        return df_clean
    
    def _validate_ohlc_relationships(self, df: pd.DataFrame) -> pd.Series:
        """
        Valida relaciones lógicas entre OHLC.
        
        Args:
            df: DataFrame con datos OHLC
            
        Returns:
            Series booleana con registros inválidos
        """
        invalid = pd.Series(False, index=df.index)
        
        # High debe ser >= Open, Close
        invalid |= (df['high'] < df['open'])
        invalid |= (df['high'] < df['close'])
        
        # Low debe ser <= Open, Close
        invalid |= (df['low'] > df['open'])
        invalid |= (df['low'] > df['close'])
        
        # High debe ser >= Low
        invalid |= (df['high'] < df['low'])
        
        return invalid
    
    def _detect_price_outliers(self, df: pd.DataFrame, 
                             threshold: float = 5.0) -> pd.Series:
        """
        Detecta outliers en precios usando z-score de retornos log.
        
        Args:
            df: DataFrame con datos de precios
            threshold: Umbral de z-score para outliers
            
        Returns:
            Series booleana con outliers detectados
        """
        if len(df) < 10:  # Necesitamos datos suficientes
            return pd.Series(False, index=df.index)
        
        # Calcular retornos log
        log_returns = np.log(df['close'] / df['close'].shift(1))
        log_returns = log_returns.dropna()
        
        if len(log_returns) < 5:
            return pd.Series(False, index=df.index)
        
        # Calcular z-scores usando MAD (más robusto que std)
        median_return = log_returns.median()
        mad = np.median(np.abs(log_returns - median_return))
        
        if mad == 0:  # Evitar división por cero
            return pd.Series(False, index=df.index)
        
        # Z-score usando MAD
        z_scores = 0.6745 * (log_returns - median_return) / mad
        
        # Detectar outliers
        outliers = np.abs(z_scores) > threshold
        
        # Expandir al índice completo del DataFrame
        outlier_mask = pd.Series(False, index=df.index)
        outlier_mask.loc[outliers.index[outliers]] = True
        
        return outlier_mask
    
    @log_execution_time("fill_missing_data")
    def fill_missing_data(self, df: pd.DataFrame, 
                         method: str = 'forward',
                         limit: Optional[int] = None) -> pd.DataFrame:
        """
        Rellena datos faltantes sin look-ahead bias.
        
        Args:
            df: DataFrame con datos
            method: Método de relleno ('forward', 'none')
            limit: Límite máximo de valores consecutivos a rellenar
            
        Returns:
            DataFrame con datos rellenados
        """
        if df.empty:
            return df
        
        df_filled = df.copy()
        
        if limit is None:
            limit = self.forward_fill_limit
        
        if method == 'forward':
            # Forward fill solo para volumen y features no causales
            # NUNCA para precios que podrían introducir look-ahead
            
            # Solo rellenar volumen con forward fill
            if 'volume' in df_filled.columns:
                before_fill = df_filled['volume'].isna().sum()
                df_filled['volume'] = df_filled['volume'].fillna(method='ffill', limit=limit)
                after_fill = df_filled['volume'].isna().sum()
                
                if before_fill > after_fill:
                    logger.info(f"Rellenados {before_fill - after_fill} valores de volumen")
            
            # Para precios, NO rellenar o usar interpolación muy conservadora
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in df_filled.columns:
                    null_count = df_filled[col].isna().sum()
                    if null_count > 0:
                        logger.warning(f"Se mantienen {null_count} valores nulos en {col} para evitar look-ahead")
        
        elif method == 'none':
            logger.info("No se aplica relleno de datos faltantes")
        
        else:
            raise ValueError(f"Método de relleno '{method}' no soportado")
        
        return df_filled
    
    def detect_and_handle_gaps(self, df: pd.DataFrame, 
                              timeframe: str) -> pd.DataFrame:
        """
        Detecta y maneja gaps en series temporales.
        
        Args:
            df: DataFrame con datos temporales
            timeframe: Timeframe esperado
            
        Returns:
            DataFrame con gaps manejados
        """
        if df.empty or len(df) < 2:
            return df
        
        # Validar continuidad
        continuity_stats = validate_data_continuity(df, timeframe, warn_threshold=0.1)
        
        # Si hay gaps grandes, reportar pero no rellenar automáticamente
        if continuity_stats['large_gaps'] > 0:
            logger.warning(f"Detectados {continuity_stats['large_gaps']} gaps grandes")
            logger.warning(f"Ubicaciones: {continuity_stats['gap_locations'][:5]}")  # Mostrar primeros 5
        
        # Para gaps pequeños, opcionalmente crear índice completo
        if self.remove_gaps and continuity_stats['small_gaps'] > 0:
            logger.info(f"Manejando {continuity_stats['small_gaps']} gaps pequeños")
            
            # Crear índice completo
            expected_freq = f"{timeframe_to_minutes(timeframe)}min"
            full_index = pd.date_range(
                start=df.index.min(),
                end=df.index.max(),
                freq=expected_freq,
                tz=df.index.tz
            )
            
            # Reindexar (esto creará NaN en gaps)
            df_reindexed = df.reindex(full_index)
            
            # NO rellenar automáticamente - mantener NaN para análisis posterior
            gap_count = df_reindexed.isna().any(axis=1).sum()
            logger.info(f"Creados {gap_count} registros NaN para gaps detectados")
            
            return df_reindexed
        
        return df
    
    @log_execution_time("create_time_splits")
    def create_time_splits(self, df: pd.DataFrame,
                          train_size: Union[float, int] = 0.7,
                          val_size: Union[float, int] = 0.15,
                          test_size: Union[float, int] = 0.15,
                          gap_size: int = 0) -> Dict[str, pd.DataFrame]:
        """
        Crea splits temporales sin fuga de información.
        
        Args:
            df: DataFrame con datos temporales
            train_size: Tamaño del conjunto de entrenamiento
            val_size: Tamaño del conjunto de validación
            test_size: Tamaño del conjunto de test
            gap_size: Separación entre conjuntos (en períodos)
            
        Returns:
            Dict con DataFrames {train, val, test}
        """
        if df.empty:
            return {'train': df, 'val': df, 'test': df}
        
        logger.info(f"Creando splits temporales para {len(df)} registros")
        
        # Validar que las proporciones sumen <= 1
        if isinstance(train_size, float) and isinstance(val_size, float) and isinstance(test_size, float):
            total_prop = train_size + val_size + test_size
            if total_prop > 1.0:
                logger.warning(f"Proporciones suman {total_prop:.2f} > 1.0, reescalando...")
                train_size = train_size / total_prop
                val_size = val_size / total_prop
                test_size = test_size / total_prop
        
        # Primer split: train + val vs test
        train_val_size = 1.0 - test_size if isinstance(test_size, float) else len(df) - test_size
        train_val_df, test_df = time_series_split_no_leakage(
            df, train_size=train_val_size, gap_size=gap_size
        )
        
        # Segundo split: train vs val
        if len(train_val_df) > 0:
            train_prop = train_size / (train_size + val_size) if isinstance(train_size, float) else train_size
            train_df, val_df = time_series_split_no_leakage(
                train_val_df, train_size=train_prop, gap_size=gap_size
            )
        else:
            train_df = val_df = train_val_df
        
        splits = {
            'train': train_df,
            'val': val_df, 
            'test': test_df
        }
        
        # Log de estadísticas
        for name, split_df in splits.items():
            if len(split_df) > 0:
                start_date = split_df.index.min().strftime('%Y-%m-%d')
                end_date = split_df.index.max().strftime('%Y-%m-%d')
                logger.info(f"{name}: {len(split_df)} registros ({start_date} a {end_date})")
            else:
                logger.warning(f"{name}: split vacío")
        
        return splits
    
    def create_walk_forward_splits(self, df: pd.DataFrame,
                                  train_window: int,
                                  test_window: int,
                                  step_size: Optional[int] = None,
                                  method: str = 'rolling') -> List[Dict[str, pd.DataFrame]]:
        """
        Crea splits de walk-forward para backtesting.
        
        Args:
            df: DataFrame con datos temporales
            train_window: Tamaño de ventana de entrenamiento (períodos)
            test_window: Tamaño de ventana de test (períodos)
            step_size: Paso entre ventanas (por defecto = test_window)
            method: 'rolling' o 'expanding'
            
        Returns:
            Lista de dicts con splits {train, test}
        """
        logger.info(f"Creando walk-forward splits: {method}, train={train_window}, test={test_window}")
        
        raw_splits = walk_forward_splits(
            df, train_window, test_window, step_size, method=method
        )
        
        # Convertir a formato dict
        formatted_splits = []
        for i, (train_df, test_df) in enumerate(raw_splits):
            split_dict = {
                'train': train_df,
                'test': test_df,
                'split_id': i,
                'train_start': train_df.index.min() if len(train_df) > 0 else None,
                'train_end': train_df.index.max() if len(train_df) > 0 else None,
                'test_start': test_df.index.min() if len(test_df) > 0 else None,
                'test_end': test_df.index.max() if len(test_df) > 0 else None
            }
            formatted_splits.append(split_dict)
        
        logger.info(f"Creados {len(formatted_splits)} splits de walk-forward")
        
        return formatted_splits
    
    @log_execution_time("preprocess_data")
    def preprocess_data(self, df: pd.DataFrame,
                       symbol: str = "unknown",
                       timeframe: str = "1h",
                       create_splits: bool = True) -> Dict[str, Any]:
        """
        Pipeline completo de preprocesamiento.
        
        Args:
            df: DataFrame con datos raw
            symbol: Símbolo para identificación
            timeframe: Timeframe de los datos
            create_splits: Si crear splits train/val/test
            
        Returns:
            Dict con datos procesados y metadata
        """
        logger.info(f"Iniciando preprocesamiento de {symbol} {timeframe}")
        
        results = {
            'raw_data': df.copy(),
            'processed_data': None,
            'splits': {},
            'validation_stats': {},
            'metadata': {
                'symbol': symbol,
                'timeframe': timeframe,
                'processing_time': None,
                'original_rows': len(df),
                'final_rows': 0,
                'removed_rows': 0
            }
        }
        
        try:
            # 1. Limpieza básica
            df_clean = self.clean_ohlcv_data(df, symbol, timeframe)
            
            # 2. Manejo de gaps
            df_gaps = self.detect_and_handle_gaps(df_clean, timeframe)
            
            # 3. Relleno conservador
            df_filled = self.fill_missing_data(df_gaps, method='forward')
            
            # 4. Validación final
            if len(df_filled) == 0:
                logger.error("No quedan datos después del preprocesamiento")
                return results
            
            results['processed_data'] = df_filled
            results['metadata']['final_rows'] = len(df_filled)
            results['metadata']['removed_rows'] = len(df) - len(df_filled)
            
            # 5. Crear splits si se requiere
            if create_splits and len(df_filled) > 10:
                splits = self.create_time_splits(df_filled)
                results['splits'] = splits
            
            # 6. Estadísticas de validación
            if f"{symbol}_{timeframe}" in self.validation_results:
                results['validation_stats'] = self.validation_results[f"{symbol}_{timeframe}"]
            
            logger.info(f"Preprocesamiento completado exitosamente para {symbol} {timeframe}")
            
        except Exception as e:
            logger.error(f"Error en preprocesamiento: {e}")
            raise
        
        return results
    
    def save_processed_data(self, processed_results: Dict[str, Any],
                           suffix: str = "") -> Dict[str, Path]:
        """
        Guarda datos procesados y splits.
        
        Args:
            processed_results: Resultados del preprocesamiento
            suffix: Sufijo para nombres de archivo
            
        Returns:
            Dict con paths de archivos guardados
        """
        metadata = processed_results['metadata']
        symbol = metadata['symbol'].replace('/', '_')
        timeframe = metadata['timeframe']
        
        base_name = f"{symbol}_{timeframe}"
        if suffix:
            base_name += f"_{suffix}"
        
        saved_files = {}
        
        # Guardar datos procesados
        if processed_results['processed_data'] is not None:
            processed_path = self.processed_data_dir / f"{base_name}_processed.parquet"
            processed_results['processed_data'].to_parquet(processed_path)
            saved_files['processed'] = processed_path
            logger.info(f"Datos procesados guardados: {processed_path}")
        
        # Guardar splits
        for split_name, split_df in processed_results['splits'].items():
            if split_df is not None and len(split_df) > 0:
                split_path = self.processed_data_dir / f"{base_name}_{split_name}.parquet"
                split_df.to_parquet(split_path)
                saved_files[f'split_{split_name}'] = split_path
        
        # Guardar metadata
        metadata_path = self.processed_data_dir / f"{base_name}_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            # Convertir timestamps para JSON
            metadata_json = metadata.copy()
            for key, value in metadata_json.items():
                if isinstance(value, pd.Timestamp):
                    metadata_json[key] = value.isoformat()
            
            json.dump({
                'metadata': metadata_json,
                'validation_stats': processed_results['validation_stats']
            }, f, indent=2)
        
        saved_files['metadata'] = metadata_path
        
        return saved_files


if __name__ == "__main__":
    # Ejemplo de uso
    from ..config_loader import load_config
    
    try:
        config = load_config("config/binance_spot_example.yaml")
        preprocessor = DataPreprocessor(config)
        
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
        
        # Introducir algunos outliers y problemas
        df.iloc[100:103] = np.nan  # Gap
        df.iloc[500, df.columns.get_loc('high')] = df.iloc[500, df.columns.get_loc('close')] * 0.5  # High < close
        
        print(f"Datos de ejemplo creados: {len(df)} registros")
        print(df.head())
        
        # Preprocesar
        results = preprocessor.preprocess_data(df, "BTC/USDT", "1h")
        
        print(f"\nResultados del preprocesamiento:")
        print(f"Registros originales: {results['metadata']['original_rows']}")
        print(f"Registros finales: {results['metadata']['final_rows']}")
        print(f"Registros removidos: {results['metadata']['removed_rows']}")
        
        for split_name, split_df in results['splits'].items():
            print(f"{split_name}: {len(split_df)} registros")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
