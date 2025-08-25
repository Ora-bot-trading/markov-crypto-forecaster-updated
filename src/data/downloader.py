"""
Descargador de datos de exchanges de criptomonedas.

Utiliza CCXT para descargar datos OHLCV de múltiples exchanges,
con manejo de rate limits, reintentos y persistencia local.
"""

import ccxt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
import time
import os
from datetime import datetime, timedelta
import json
import warnings

from ..config_loader import Config
from ..logging_utils import get_logger, log_execution_time
from ..utils_time import parse_datetime, timeframe_to_minutes

logger = get_logger(__name__)


class ExchangeDataDownloader:
    """
    Descargador de datos de exchanges usando CCXT.
    
    Soporta múltiples exchanges con rate limiting automático,
    reintentos exponenciales y cache local.
    """
    
    def __init__(self, config: Union[Config, dict]):
        """
        Inicializa el descargador.
        
        Args:
            config: Configuración del sistema
        """
        if isinstance(config, dict):
            self.config = config
        else:
            self.config = config.dict()
            
        self.data_config = self.config.get('data', {})
        self.api_config = self.config.get('api', {})
        self.paths_config = self.config.get('paths', {})
        
        self.exchange_name = self.data_config.get('exchange', 'binance')
        self.rate_limit_sleep = self.data_config.get('rate_limit_sleep', 1.0)
        self.max_retries = 3
        self.retry_delay = 2.0
        
        self._exchange = None
        self._last_request_time = 0
        
        # Crear directorios
        self.raw_data_dir = Path(self.paths_config.get('data_dir', 'data')) / 'raw'
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def exchange(self) -> ccxt.Exchange:
        """
        Propiedad lazy para obtener la instancia del exchange.
        
        Returns:
            Instancia configurada del exchange
        """
        if self._exchange is None:
            self._exchange = self._create_exchange()
        return self._exchange
    
    def _create_exchange(self) -> ccxt.Exchange:
        """
        Crea y configura la instancia del exchange.
        
        Returns:
            Exchange configurado
            
        Raises:
            ValueError: Si el exchange no es soportado
        """
        try:
            # Obtener clase del exchange
            exchange_class = getattr(ccxt, self.exchange_name)
        except AttributeError:
            raise ValueError(f"Exchange '{self.exchange_name}' no soportado por CCXT")
        
        # Configuración base
        exchange_config = {
            'timeout': 30000,  # 30 segundos
            'enableRateLimit': True,
            'sandbox': False
        }
        
        # API keys si están disponibles
        api_keys = self.api_config.get(self.exchange_name, {})
        if 'binance_api_key' in api_keys:
            exchange_config['apiKey'] = api_keys['binance_api_key']
        if 'binance_api_secret' in api_keys:
            exchange_config['secret'] = api_keys['binance_api_secret']
        if 'binance_testnet' in api_keys:
            exchange_config['sandbox'] = api_keys['binance_testnet'].lower() == 'true'
        
        # Configuración específica del exchange
        exchange_params = self.data_config.get('exchange_params', {})
        exchange_config.update(exchange_params)
        
        # Crear instancia
        exchange = exchange_class(exchange_config)
        
        # Cargar mercados
        try:
            exchange.load_markets()
            logger.info(f"Exchange {self.exchange_name} configurado exitosamente")
        except Exception as e:
            logger.warning(f"No se pudieron cargar mercados: {e}")
        
        return exchange
    
    def _rate_limit_delay(self):
        """Aplica rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self.rate_limit_sleep:
            sleep_time = self.rate_limit_sleep - time_since_last
            time.sleep(sleep_time)
        
        self._last_request_time = time.time()
    
    def _retry_request(self, func, *args, **kwargs):
        """
        Ejecuta una función con reintentos exponenciales.
        
        Args:
            func: Función a ejecutar
            *args, **kwargs: Argumentos para la función
            
        Returns:
            Resultado de la función
            
        Raises:
            Exception: Si todos los reintentos fallan
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                self._rate_limit_delay()
                return func(*args, **kwargs)
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as e:
                last_exception = e
                delay = self.retry_delay * (2 ** attempt)
                logger.warning(f"Intento {attempt + 1} falló: {e}. Reintentando en {delay}s...")
                time.sleep(delay)
            except ccxt.RateLimitExceeded as e:
                last_exception = e
                delay = self.retry_delay * (2 ** attempt) * 2  # Delay más largo para rate limits
                logger.warning(f"Rate limit excedido. Esperando {delay}s...")
                time.sleep(delay)
            except Exception as e:
                # Para otros errores, no reintentar
                logger.error(f"Error no recuperable: {e}")
                raise
        
        # Si llegamos aquí, todos los reintentos fallaron
        logger.error(f"Todos los reintentos fallaron. Último error: {last_exception}")
        raise last_exception
    
    def get_available_symbols(self) -> List[str]:
        """
        Obtiene lista de símbolos disponibles en el exchange.
        
        Returns:
            Lista de símbolos disponibles
        """
        try:
            markets = self.exchange.markets
            return sorted(list(markets.keys()))
        except Exception as e:
            logger.error(f"Error al obtener símbolos: {e}")
            return []
    
    def validate_symbol_timeframe(self, symbol: str, timeframe: str) -> bool:
        """
        Valida si un símbolo y timeframe están disponibles.
        
        Args:
            symbol: Símbolo a validar (ej: 'BTC/USDT')
            timeframe: Timeframe a validar (ej: '1h')
            
        Returns:
            True si están disponibles
        """
        try:
            # Verificar símbolo
            if symbol not in self.exchange.markets:
                logger.warning(f"Símbolo '{symbol}' no disponible en {self.exchange_name}")
                return False
            
            # Verificar timeframe
            if hasattr(self.exchange, 'timeframes') and timeframe not in self.exchange.timeframes:
                logger.warning(f"Timeframe '{timeframe}' no disponible en {self.exchange_name}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error al validar símbolo/timeframe: {e}")
            return False
    
    @log_execution_time("download_ohlcv")
    def download_ohlcv(self, 
                      symbol: str,
                      timeframe: str,
                      start_date: Optional[Union[str, datetime]] = None,
                      end_date: Optional[Union[str, datetime]] = None,
                      limit: Optional[int] = None) -> pd.DataFrame:
        """
        Descarga datos OHLCV de un símbolo.
        
        Args:
            symbol: Símbolo a descargar (ej: 'BTC/USDT')
            timeframe: Timeframe (ej: '1h', '1d')
            start_date: Fecha de inicio (opcional)
            end_date: Fecha de fin (opcional)
            limit: Límite de velas (opcional)
            
        Returns:
            DataFrame con datos OHLCV
            
        Raises:
            ValueError: Si parámetros no válidos
            Exception: Si error en descarga
        """
        # Validar parámetros
        if not self.validate_symbol_timeframe(symbol, timeframe):
            raise ValueError(f"Símbolo '{symbol}' o timeframe '{timeframe}' no válidos")
        
        # Parsear fechas
        start_ts = None
        end_ts = None
        
        if start_date:
            start_dt = parse_datetime(start_date)
            start_ts = int(start_dt.timestamp() * 1000)  # CCXT usa millisegundos
        
        if end_date:
            end_dt = parse_datetime(end_date)
            end_ts = int(end_dt.timestamp() * 1000)
        
        # Usar límite de configuración si no se especifica
        if limit is None:
            limit = self.data_config.get('max_candles', 1000)
        
        logger.info(f"Descargando {symbol} {timeframe} desde {start_date} hasta {end_date}")
        
        # Descargar datos
        all_data = []
        current_start = start_ts
        
        while True:
            try:
                # Determinar parámetros para esta iteración
                fetch_params = {}
                if current_start:
                    fetch_params['since'] = current_start
                
                # Límite efectivo
                effective_limit = limit
                if end_ts and current_start:
                    # Calcular máximo número de velas posibles hasta end_date
                    timeframe_ms = timeframe_to_minutes(timeframe) * 60 * 1000
                    max_possible = int((end_ts - current_start) / timeframe_ms)
                    effective_limit = min(limit, max_possible)
                    
                    if effective_limit <= 0:
                        break
                
                # Realizar descarga
                ohlcv = self._retry_request(
                    self.exchange.fetch_ohlcv,
                    symbol,
                    timeframe,
                    limit=effective_limit,
                    **fetch_params
                )
                
                if not ohlcv:
                    logger.warning("No se obtuvieron datos")
                    break
                
                all_data.extend(ohlcv)
                logger.debug(f"Descargadas {len(ohlcv)} velas. Total: {len(all_data)}")
                
                # Verificar si hemos llegado al final
                last_timestamp = ohlcv[-1][0]
                
                if end_ts and last_timestamp >= end_ts:
                    break
                
                if len(ohlcv) < effective_limit:
                    # No hay más datos disponibles
                    break
                
                # Preparar siguiente iteración
                current_start = last_timestamp + timeframe_to_minutes(timeframe) * 60 * 1000
                
            except Exception as e:
                logger.error(f"Error en descarga: {e}")
                raise
        
        if not all_data:
            logger.warning("No se descargaron datos")
            return pd.DataFrame()
        
        # Convertir a DataFrame
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Procesar timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        
        # Convertir tipos
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_columns] = df[numeric_columns].astype(float)
        
        # Filtrar por rango de fechas si es necesario
        if start_date:
            start_dt = parse_datetime(start_date)
            df = df[df.index >= start_dt]
        
        if end_date:
            end_dt = parse_datetime(end_date)
            df = df[df.index <= end_dt]
        
        # Remover duplicados y ordenar
        df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index()
        
        logger.info(f"Descarga completada: {len(df)} velas de {symbol} {timeframe}")
        
        return df
    
    def save_data(self, df: pd.DataFrame, 
                 symbol: str, 
                 timeframe: str,
                 format: str = 'parquet') -> Path:
        """
        Guarda datos en el filesystem local.
        
        Args:
            df: DataFrame a guardar
            symbol: Símbolo de los datos
            timeframe: Timeframe de los datos
            format: Formato de archivo ('parquet', 'csv')
            
        Returns:
            Path del archivo guardado
        """
        # Crear nombre de archivo
        safe_symbol = symbol.replace('/', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'parquet':
            filename = f"{safe_symbol}_{timeframe}_{timestamp}.parquet"
            file_path = self.raw_data_dir / filename
            df.to_parquet(file_path)
        elif format == 'csv':
            filename = f"{safe_symbol}_{timeframe}_{timestamp}.csv"
            file_path = self.raw_data_dir / filename
            df.to_csv(file_path)
        else:
            raise ValueError(f"Formato '{format}' no soportado")
        
        logger.info(f"Datos guardados en: {file_path}")
        
        # Guardar metadatos
        metadata = {
            'symbol': symbol,
            'timeframe': timeframe,
            'exchange': self.exchange_name,
            'download_time': timestamp,
            'rows': len(df),
            'start_date': df.index.min().isoformat() if len(df) > 0 else None,
            'end_date': df.index.max().isoformat() if len(df) > 0 else None
        }
        
        metadata_path = file_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return file_path
    
    def load_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Carga datos desde archivo local.
        
        Args:
            file_path: Ruta al archivo
            
        Returns:
            DataFrame con los datos
        """
        file_path = Path(file_path)
        
        if file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        elif file_path.suffix == '.csv':
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        else:
            raise ValueError(f"Formato de archivo no soportado: {file_path.suffix}")
        
        logger.info(f"Datos cargados desde: {file_path}")
        return df
    
    def get_latest_data_file(self, symbol: str, timeframe: str) -> Optional[Path]:
        """
        Obtiene el archivo de datos más reciente para un símbolo/timeframe.
        
        Args:
            symbol: Símbolo a buscar
            timeframe: Timeframe a buscar
            
        Returns:
            Path del archivo más reciente o None si no existe
        """
        safe_symbol = symbol.replace('/', '_')
        pattern = f"{safe_symbol}_{timeframe}_*.parquet"
        
        matching_files = list(self.raw_data_dir.glob(pattern))
        
        if not matching_files:
            pattern = f"{safe_symbol}_{timeframe}_*.csv"
            matching_files = list(self.raw_data_dir.glob(pattern))
        
        if not matching_files:
            return None
        
        # Obtener el más reciente por fecha de modificación
        latest_file = max(matching_files, key=lambda p: p.stat().st_mtime)
        return latest_file
    
    @log_execution_time("download_and_save")
    def download_and_save(self, 
                         symbol: Optional[str] = None,
                         timeframe: Optional[str] = None,
                         start_date: Optional[Union[str, datetime]] = None,
                         end_date: Optional[Union[str, datetime]] = None,
                         overwrite: bool = False) -> Path:
        """
        Descarga y guarda datos usando configuración.
        
        Args:
            symbol: Símbolo (usa config si None)
            timeframe: Timeframe (usa config si None)
            start_date: Fecha inicio (usa config si None)
            end_date: Fecha fin (usa config si None)
            overwrite: Si sobrescribir archivos existentes
            
        Returns:
            Path del archivo guardado
        """
        # Usar configuración si no se especifican parámetros
        symbol = symbol or self.data_config.get('symbol', 'BTC/USDT')
        timeframe = timeframe or self.data_config.get('timeframe', '1h')
        start_date = start_date or self.data_config.get('start_date')
        end_date = end_date or self.data_config.get('end_date')
        
        # Verificar si ya existe archivo reciente
        if not overwrite:
            existing_file = self.get_latest_data_file(symbol, timeframe)
            if existing_file:
                # Verificar si es reciente (menos de 1 día)
                file_age = time.time() - existing_file.stat().st_mtime
                if file_age < 86400:  # 24 horas
                    logger.info(f"Usando archivo existente: {existing_file}")
                    return existing_file
        
        # Descargar datos
        df = self.download_ohlcv(symbol, timeframe, start_date, end_date)
        
        if df.empty:
            raise ValueError("No se obtuvieron datos para descargar")
        
        # Guardar datos
        file_path = self.save_data(df, symbol, timeframe)
        
        return file_path


def download_multiple_symbols(config: Union[Config, dict],
                            symbols: List[str],
                            timeframes: List[str] = None,
                            parallel: bool = False) -> Dict[str, Dict[str, Path]]:
    """
    Descarga datos para múltiples símbolos.
    
    Args:
        config: Configuración del sistema
        symbols: Lista de símbolos a descargar
        timeframes: Lista de timeframes (usa config si None)
        parallel: Si descargar en paralelo (no implementado)
        
    Returns:
        Dict con paths de archivos {symbol: {timeframe: path}}
    """
    if timeframes is None:
        if isinstance(config, dict):
            timeframes = [config.get('data', {}).get('timeframe', '1h')]
        else:
            timeframes = [config.data.timeframe]
    
    results = {}
    
    for symbol in symbols:
        results[symbol] = {}
        downloader = ExchangeDataDownloader(config)
        
        for timeframe in timeframes:
            try:
                logger.info(f"Descargando {symbol} {timeframe}")
                file_path = downloader.download_and_save(symbol=symbol, timeframe=timeframe)
                results[symbol][timeframe] = file_path
            except Exception as e:
                logger.error(f"Error descargando {symbol} {timeframe}: {e}")
                results[symbol][timeframe] = None
    
    return results


if __name__ == "__main__":
    # Ejemplo de uso
    from ..config_loader import load_config
    
    try:
        config = load_config("config/binance_spot_example.yaml")
        downloader = ExchangeDataDownloader(config)
        
        # Descargar datos
        df = downloader.download_ohlcv('BTC/USDT', '1h', limit=100)
        print(f"Descargados {len(df)} registros")
        print(df.head())
        
        # Guardar datos
        file_path = downloader.save_data(df, 'BTC/USDT', '1h')
        print(f"Guardado en: {file_path}")
        
    except Exception as e:
        print(f"Error: {e}")
