"""
Utilidades de logging para markov-crypto-forecaster.

Configura el sistema de logging usando loguru con rotación de archivos,
niveles configurables y formateo personalizado.
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger
import logging


def setup_logging(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Configura el sistema de logging usando loguru.
    
    Args:
        config: Configuración de logging. Si es None, usa configuración por defecto.
                Debe contener keys: level, format, file, rotation, retention
    """
    # Configuración por defecto
    default_config = {
        "level": "INFO",
        "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
        "file": "logs/markov_forecaster.log",
        "rotation": "1 week",
        "retention": "1 month"
    }
    
    if config is None:
        config = default_config
    else:
        # Fusionar con defaults
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
    
    # Remover handlers existentes
    logger.remove()
    
    # Handler para consola
    logger.add(
        sys.stderr,
        level=config["level"],
        format=config["format"],
        colorize=True,
        enqueue=True
    )
    
    # Handler para archivo (si se especifica)
    if config.get("file"):
        log_file = Path(config["file"])
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            str(log_file),
            level=config["level"],
            format=config["format"],
            rotation=config.get("rotation", "1 week"),
            retention=config.get("retention", "1 month"),
            compression="zip",
            enqueue=True
        )
    
    # Configurar interceptor para logging estándar de Python
    # Esto redirige logs de otras librerías (como requests, urllib3, etc.) a loguru
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            # Get corresponding Loguru level if it exists
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # Find caller from where originated the logged message
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

    # Configurar intercepción para loggers comunes
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Configurar niveles específicos para librerías externas
    external_loggers = [
        "urllib3.connectionpool",
        "requests.packages.urllib3",
        "ccxt",
        "binance",
        "matplotlib",
        "plotly"
    ]
    
    for logger_name in external_loggers:
        external_logger = logging.getLogger(logger_name)
        external_logger.setLevel(logging.WARNING)
    
    logger.info("Sistema de logging configurado exitosamente")


def get_logger(name: str) -> "logger":
    """
    Obtiene un logger con un nombre específico.
    
    Args:
        name: Nombre del logger (normalmente __name__)
        
    Returns:
        Logger configurado
    """
    return logger.bind(name=name)


def log_execution_time(func_name: str = None):
    """
    Decorador para loggear tiempo de ejecución de funciones.
    
    Args:
        func_name: Nombre personalizado para la función (opcional)
    """
    def decorator(func):
        import functools
        import time
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = func_name or func.__name__
            start_time = time.time()
            
            logger.debug(f"Iniciando ejecución de {name}")
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(f"{name} completado en {execution_time:.2f} segundos")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{name} falló después de {execution_time:.2f} segundos: {e}")
                raise
                
        return wrapper
    return decorator


def log_function_call(include_args: bool = False, include_result: bool = False):
    """
    Decorador para loggear llamadas a funciones.
    
    Args:
        include_args: Si incluir argumentos en el log
        include_result: Si incluir el resultado en el log
    """
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            
            if include_args:
                args_str = f"args={args}, kwargs={kwargs}"
                logger.debug(f"Llamando {func_name} con {args_str}")
            else:
                logger.debug(f"Llamando {func_name}")
            
            try:
                result = func(*args, **kwargs)
                
                if include_result:
                    logger.debug(f"{func_name} retornó: {result}")
                else:
                    logger.debug(f"{func_name} completado exitosamente")
                    
                return result
            except Exception as e:
                logger.error(f"{func_name} falló: {e}")
                raise
                
        return wrapper
    return decorator


def create_run_logger(run_id: str, base_path: str = "logs") -> "logger":
    """
    Crea un logger específico para un run de entrenamiento/backtest.
    
    Args:
        run_id: Identificador único del run
        base_path: Directorio base para logs
        
    Returns:
        Logger configurado para este run específico
    """
    log_path = Path(base_path) / f"run_{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Crear un logger específico para este run
    run_logger = logger.bind(run_id=run_id)
    
    # Agregar handler específico para este run
    logger.add(
        str(log_path),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | RUN:{extra[run_id]} | {name}:{function}:{line} - {message}",
        filter=lambda record: record["extra"].get("run_id") == run_id,
        rotation="100 MB",
        compression="zip"
    )
    
    run_logger.info(f"Logger de run {run_id} iniciado")
    return run_logger


def setup_quiet_mode():
    """
    Configura modo silencioso (solo errores).
    """
    logger.remove()
    logger.add(sys.stderr, level="ERROR", format="{level} | {message}")


def setup_verbose_mode():
    """
    Configura modo verbose (debug level).
    """
    logger.remove()
    logger.add(
        sys.stderr, 
        level="DEBUG", 
        format="{time:HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
        colorize=True
    )


class LoggerContextManager:
    """
    Context manager para logging temporal con configuración específica.
    """
    
    def __init__(self, level: str = "INFO", file: Optional[str] = None):
        self.level = level
        self.file = file
        self.handler_id = None
        
    def __enter__(self):
        if self.file:
            self.handler_id = logger.add(
                self.file,
                level=self.level,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
            )
        return logger
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handler_id:
            logger.remove(self.handler_id)


# Configurar interceptor para warnings
import warnings

def warning_handler(message, category, filename, lineno, file=None, line=None):
    """Handler personalizado para warnings."""
    logger.warning(f"{category.__name__}: {message} ({filename}:{lineno})")

# Redirigir warnings a loguru
warnings.showwarning = warning_handler


if __name__ == "__main__":
    # Ejemplo de uso
    setup_logging()
    
    test_logger = get_logger(__name__)
    test_logger.info("Test de logging")
    test_logger.debug("Mensaje de debug")
    test_logger.warning("Mensaje de warning")
    
    # Test del decorador
    @log_execution_time("test_function")
    def test_function():
        import time
        time.sleep(0.1)
        return "resultado"
    
    result = test_function()
    test_logger.info(f"Resultado: {result}")
