"""
Utilidades de manejo de tiempo para markov-crypto-forecaster.

Funciones para conversión de timestamps, splits temporales sin fuga,
validación de fechas y manejo de timezone-aware datetimes.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Union, Tuple, List, Optional
import pytz
from dateutil import parser
import ccxt
import warnings


def parse_datetime(dt_input: Union[str, datetime, date, pd.Timestamp, int, float, None]) -> Optional[pd.Timestamp]:
    """
    Convierte diferentes formatos de fecha/hora a pd.Timestamp timezone-aware.
    
    Args:
        dt_input: Fecha en cualquier formato soportado
        
    Returns:
        pd.Timestamp timezone-aware (UTC) o None si input es None
        
    Raises:
        ValueError: Si no se puede parsear la fecha
    """
    if dt_input is None:
        return None
    
    try:
        # Si es timestamp Unix (int/float)
        if isinstance(dt_input, (int, float)):
            # Detectar si es en segundos o milisegundos
            if dt_input > 1e10:  # milisegundos
                dt_input = dt_input / 1000
            return pd.Timestamp.fromtimestamp(dt_input, tz='UTC')
        
        # Si es string
        if isinstance(dt_input, str):
            # Intentar parsear con dateutil (muy flexible)
            parsed = parser.parse(dt_input)
            if parsed.tzinfo is None:
                # Asumir UTC si no hay timezone
                parsed = parsed.replace(tzinfo=pytz.UTC)
            return pd.Timestamp(parsed).tz_convert('UTC')
        
        # Si es date (sin hora)
        if isinstance(dt_input, date) and not isinstance(dt_input, datetime):
            dt_input = datetime.combine(dt_input, datetime.min.time())
        
        # Si es datetime
        if isinstance(dt_input, datetime):
            if dt_input.tzinfo is None:
                dt_input = dt_input.replace(tzinfo=pytz.UTC)
            return pd.Timestamp(dt_input).tz_convert('UTC')
        
        # Si ya es pd.Timestamp
        if isinstance(dt_input, pd.Timestamp):
            if dt_input.tz is None:
                return dt_input.tz_localize('UTC')
            else:
                return dt_input.tz_convert('UTC')
        
        # Último intento: usar pd.to_datetime
        result = pd.to_datetime(dt_input, utc=True)
        return result
        
    except Exception as e:
        raise ValueError(f"No se pudo parsear fecha '{dt_input}': {e}")


def timeframe_to_minutes(timeframe: str) -> int:
    """
    Convierte un timeframe string a minutos.
    
    Args:
        timeframe: Timeframe en formato CCXT (ej: '1m', '5m', '1h', '1d')
        
    Returns:
        Número de minutos
        
    Raises:
        ValueError: Si el timeframe no es válido
    """
    timeframe_minutes = {
        '1m': 1,
        '3m': 3,
        '5m': 5,
        '15m': 15,
        '30m': 30,
        '1h': 60,
        '2h': 120,
        '4h': 240,
        '6h': 360,
        '8h': 480,
        '12h': 720,
        '1d': 1440,
        '3d': 4320,
        '1w': 10080,
        '1M': 43200  # Aproximado
    }
    
    if timeframe not in timeframe_minutes:
        raise ValueError(f"Timeframe '{timeframe}' no soportado. Válidos: {list(timeframe_minutes.keys())}")
    
    return timeframe_minutes[timeframe]


def timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    """
    Convierte un timeframe string a pd.Timedelta.
    
    Args:
        timeframe: Timeframe en formato CCXT
        
    Returns:
        pd.Timedelta correspondiente
    """
    minutes = timeframe_to_minutes(timeframe)
    return pd.Timedelta(minutes=minutes)


def get_market_sessions_utc() -> dict:
    """
    Retorna horarios de sesiones de mercado principales en UTC.
    
    Returns:
        Dict con horarios de sesiones {session_name: (start_hour, end_hour)}
    """
    return {
        'asian': (0, 9),        # 0:00 - 9:00 UTC
        'european': (7, 16),    # 7:00 - 16:00 UTC  
        'american': (13, 22),   # 13:00 - 22:00 UTC
        'crypto': (0, 24)       # 24/7
    }


def is_market_hours(timestamp: pd.Timestamp, session: str = 'crypto') -> bool:
    """
    Verifica si un timestamp está dentro de horarios de mercado.
    
    Args:
        timestamp: Timestamp a verificar (debe ser timezone-aware)
        session: Sesión de mercado ('asian', 'european', 'american', 'crypto')
        
    Returns:
        True si está en horarios de mercado
    """
    sessions = get_market_sessions_utc()
    
    if session not in sessions:
        raise ValueError(f"Sesión '{session}' no válida. Válidas: {list(sessions.keys())}")
    
    start_hour, end_hour = sessions[session]
    
    # Convertir a UTC si no lo está
    if timestamp.tz is None:
        timestamp = timestamp.tz_localize('UTC')
    else:
        timestamp = timestamp.tz_convert('UTC')
    
    hour = timestamp.hour
    
    if start_hour <= end_hour:
        return start_hour <= hour < end_hour
    else:  # Sesión que cruza medianoche
        return hour >= start_hour or hour < end_hour


def time_series_split_no_leakage(df: pd.DataFrame, 
                                 train_size: Union[int, float], 
                                 test_size: Union[int, float, None] = None,
                                 gap_size: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split temporal sin fuga de información futura.
    
    Args:
        df: DataFrame con índice temporal
        train_size: Tamaño del conjunto de entrenamiento (int=filas, float=proporción)
        test_size: Tamaño del conjunto de test (opcional)
        gap_size: Número de períodos de separación entre train y test
        
    Returns:
        Tupla (train_df, test_df)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame debe tener índice temporal (DatetimeIndex)")
    
    n_samples = len(df)
    
    # Convertir train_size a número de filas
    if isinstance(train_size, float):
        if not 0 < train_size < 1:
            raise ValueError("train_size como float debe estar entre 0 y 1")
        train_end_idx = int(n_samples * train_size)
    else:
        train_end_idx = min(train_size, n_samples)
    
    # Índice de inicio del test (considerando gap)
    test_start_idx = train_end_idx + gap_size
    
    if test_start_idx >= n_samples:
        raise ValueError("No hay suficientes datos para crear conjunto de test después del gap")
    
    # Convertir test_size a número de filas
    if test_size is None:
        test_end_idx = n_samples
    elif isinstance(test_size, float):
        if not 0 < test_size < 1:
            raise ValueError("test_size como float debe estar entre 0 y 1")
        test_end_idx = min(test_start_idx + int(n_samples * test_size), n_samples)
    else:
        test_end_idx = min(test_start_idx + test_size, n_samples)
    
    # Crear splits
    train_df = df.iloc[:train_end_idx].copy()
    test_df = df.iloc[test_start_idx:test_end_idx].copy()
    
    return train_df, test_df


def walk_forward_splits(df: pd.DataFrame,
                       train_window: int,
                       test_window: int,
                       step_size: int = None,
                       gap_size: int = 0,
                       method: str = 'rolling') -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Genera splits de walk-forward para backtesting.
    
    Args:
        df: DataFrame con índice temporal
        train_window: Tamaño de la ventana de entrenamiento (en períodos)
        test_window: Tamaño de la ventana de test (en períodos)
        step_size: Tamaño del paso entre splits (por defecto = test_window)
        gap_size: Separación entre train y test
        method: 'rolling' (ventana móvil) o 'expanding' (ventana expansiva)
        
    Returns:
        Lista de tuplas (train_df, test_df)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame debe tener índice temporal")
    
    if method not in ['rolling', 'expanding']:
        raise ValueError("method debe ser 'rolling' o 'expanding'")
    
    if step_size is None:
        step_size = test_window
    
    n_samples = len(df)
    splits = []
    
    start_idx = 0
    
    while True:
        if method == 'rolling':
            train_start_idx = start_idx
            train_end_idx = start_idx + train_window
        else:  # expanding
            train_start_idx = 0
            train_end_idx = start_idx + train_window
        
        test_start_idx = train_end_idx + gap_size
        test_end_idx = test_start_idx + test_window
        
        # Verificar que hay suficientes datos
        if test_end_idx > n_samples:
            break
        
        # Crear splits
        train_df = df.iloc[train_start_idx:train_end_idx].copy()
        test_df = df.iloc[test_start_idx:test_end_idx].copy()
        
        splits.append((train_df, test_df))
        
        # Avanzar
        start_idx += step_size
    
    return splits


def resample_ohlcv(df: pd.DataFrame, 
                   timeframe: str,
                   agg_dict: dict = None) -> pd.DataFrame:
    """
    Resamplea datos OHLCV a un timeframe diferente.
    
    Args:
        df: DataFrame con columnas OHLCV
        timeframe: Nuevo timeframe (ej: '4h', '1d')
        agg_dict: Diccionario de agregación personalizado
        
    Returns:
        DataFrame resampled
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame debe tener índice temporal")
    
    # Mapeo por defecto para OHLCV
    default_agg = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    # Detectar columnas OHLCV (case insensitive)
    columns_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ['open', 'high', 'low', 'close', 'volume']:
            columns_map[col] = default_agg[col_lower]
        elif col_lower in ['vol', 'v']:
            columns_map[col] = 'sum'
        elif col_lower in ['o', 'h', 'l', 'c']:
            ohlc_map = {'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last'}
            columns_map[col] = ohlc_map[col_lower]
    
    # Usar agregación personalizada si se proporciona
    if agg_dict:
        columns_map.update(agg_dict)
    
    # Agregar columnas faltantes con 'last' por defecto
    for col in df.columns:
        if col not in columns_map:
            columns_map[col] = 'last'
    
    # Convertir timeframe a offset de pandas
    offset_map = {
        '1m': '1T', '3m': '3T', '5m': '5T', '15m': '15T', '30m': '30T',
        '1h': '1H', '2h': '2H', '4h': '4H', '6h': '6H', '8h': '8H', '12h': '12H',
        '1d': '1D', '3d': '3D', '1w': '1W', '1M': '1M'
    }
    
    if timeframe not in offset_map:
        raise ValueError(f"Timeframe '{timeframe}' no soportado")
    
    offset = offset_map[timeframe]
    
    # Resamplear
    resampled = df.resample(offset, label='left', closed='left').agg(columns_map)
    
    # Eliminar filas con NaN (períodos incompletos)
    resampled = resampled.dropna()
    
    return resampled


def validate_data_continuity(df: pd.DataFrame, 
                           timeframe: str,
                           max_gap_multiplier: float = 3.0,
                           warn_threshold: float = 0.05) -> dict:
    """
    Valida la continuidad de datos temporales.
    
    Args:
        df: DataFrame con índice temporal
        timeframe: Timeframe esperado
        max_gap_multiplier: Multiplicador para detectar gaps grandes
        warn_threshold: Umbral de % gaps para warning
        
    Returns:
        Dict con estadísticas de continuidad
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame debe tener índice temporal")
    
    # Calcular diferencias de tiempo
    time_diffs = df.index.to_series().diff()[1:]
    expected_diff = timeframe_to_timedelta(timeframe)
    
    # Detectar gaps
    normal_gaps = time_diffs == expected_diff
    small_gaps = (time_diffs > expected_diff) & (time_diffs <= expected_diff * 2)
    large_gaps = time_diffs > expected_diff * max_gap_multiplier
    
    # Estadísticas
    stats = {
        'total_periods': len(df),
        'expected_periods': int((df.index[-1] - df.index[0]) / expected_diff) + 1,
        'normal_gaps': normal_gaps.sum(),
        'small_gaps': small_gaps.sum(), 
        'large_gaps': large_gaps.sum(),
        'missing_periods': 0,
        'continuity_score': 0.0,
        'largest_gap': time_diffs.max(),
        'gap_locations': []
    }
    
    stats['missing_periods'] = stats['expected_periods'] - stats['total_periods']
    stats['continuity_score'] = stats['normal_gaps'] / len(time_diffs) if len(time_diffs) > 0 else 1.0
    
    # Ubicaciones de gaps grandes
    if large_gaps.any():
        gap_indices = df.index[1:][large_gaps]
        stats['gap_locations'] = gap_indices.tolist()
    
    # Warnings
    gap_percentage = (stats['small_gaps'] + stats['large_gaps']) / len(time_diffs) if len(time_diffs) > 0 else 0
    
    if gap_percentage > warn_threshold:
        warnings.warn(f"Alta proporción de gaps: {gap_percentage:.2%}")
    
    if stats['large_gaps'] > 0:
        warnings.warn(f"Detectados {stats['large_gaps']} gaps grandes")
    
    return stats


def generate_trading_calendar(start_date: Union[str, datetime],
                            end_date: Union[str, datetime],
                            timeframe: str = '1d',
                            market_type: str = 'crypto') -> pd.DatetimeIndex:
    """
    Genera un calendario de trading.
    
    Args:
        start_date: Fecha de inicio
        end_date: Fecha de fin
        timeframe: Timeframe para el calendario
        market_type: 'crypto' (24/7) o 'traditional' (días laborables)
        
    Returns:
        DatetimeIndex con fechas de trading
    """
    start_dt = parse_datetime(start_date)
    end_dt = parse_datetime(end_date)
    
    freq_map = {
        '1m': 'T', '5m': '5T', '15m': '15T', '30m': '30T',
        '1h': 'H', '4h': '4H', '1d': 'D', '1w': 'W'
    }
    
    if timeframe not in freq_map:
        raise ValueError(f"Timeframe '{timeframe}' no soportado para calendario")
    
    # Generar rango completo
    calendar = pd.date_range(start=start_dt, end=end_dt, freq=freq_map[timeframe], tz='UTC')
    
    # Filtrar por tipo de mercado
    if market_type == 'traditional':
        # Excluir fines de semana
        calendar = calendar[calendar.dayofweek < 5]  # Lunes=0, Viernes=4
    elif market_type == 'crypto':
        # Crypto es 24/7, no filtrar
        pass
    else:
        raise ValueError("market_type debe ser 'crypto' o 'traditional'")
    
    return calendar


def align_dataframes(*dfs: pd.DataFrame, 
                    method: str = 'inner',
                    fill_method: str = None) -> List[pd.DataFrame]:
    """
    Alinea múltiples DataFrames por índice temporal.
    
    Args:
        *dfs: DataFrames a alinear
        method: 'inner', 'outer', 'left', 'right'
        fill_method: 'ffill', 'bfill', None
        
    Returns:
        Lista de DataFrames alineados
    """
    if not dfs:
        return []
    
    if len(dfs) == 1:
        return list(dfs)
    
    # Verificar que todos tienen índice temporal
    for i, df in enumerate(dfs):
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"DataFrame {i} debe tener índice temporal")
    
    # Crear índice unificado
    if method == 'inner':
        unified_index = dfs[0].index
        for df in dfs[1:]:
            unified_index = unified_index.intersection(df.index)
    elif method == 'outer':
        unified_index = dfs[0].index
        for df in dfs[1:]:
            unified_index = unified_index.union(df.index)
    elif method == 'left':
        unified_index = dfs[0].index
    elif method == 'right':
        unified_index = dfs[-1].index
    else:
        raise ValueError("method debe ser 'inner', 'outer', 'left' o 'right'")
    
    # Alinear DataFrames
    aligned = []
    for df in dfs:
        aligned_df = df.reindex(unified_index)
        
        if fill_method:
            aligned_df = aligned_df.fillna(method=fill_method)
        
        aligned.append(aligned_df)
    
    return aligned


if __name__ == "__main__":
    # Ejemplos de uso
    
    # Test de parsing
    dates = ['2023-01-01', '2023-01-01 12:00:00', 1672531200, datetime.now()]
    for dt in dates:
        parsed = parse_datetime(dt)
        print(f"{dt} -> {parsed}")
    
    # Test de timeframe
    print(f"1h = {timeframe_to_minutes('1h')} minutos")
    print(f"1d = {timeframe_to_minutes('1d')} minutos")
    
    # Test de sesiones de mercado
    now = pd.Timestamp.now(tz='UTC')
    print(f"Es horario crypto: {is_market_hours(now, 'crypto')}")
    print(f"Es horario europeo: {is_market_hours(now, 'european')}")
