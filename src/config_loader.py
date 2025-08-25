"""
Cargador de configuración para markov-crypto-forecaster.

Maneja la carga y validación de archivos de configuración YAML,
herencia de configuraciones y validación con Pydantic.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from pydantic import BaseModel, Field, validator
from datetime import datetime, date
import logging

logger = logging.getLogger(__name__)


class DataConfig(BaseModel):
    """Configuración de datos y descarga."""
    exchange: str = "binance"
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    start_date: Union[str, date, None] = None
    end_date: Union[str, date, None] = None
    max_candles: int = 1000
    rate_limit_sleep: float = 1.0
    
    @validator('timeframe')
    def validate_timeframe(cls, v):
        valid_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
        if v not in valid_timeframes:
            raise ValueError(f'Timeframe debe ser uno de: {valid_timeframes}')
        return v


class FeatureConfig(BaseModel):
    """Configuración de features."""
    enabled: List[str] = Field(default_factory=lambda: [
        "log_returns", "volatility", "atr", "volume_delta", "realized_vol", "z_scores"
    ])
    params: Dict[str, Any] = Field(default_factory=dict)
    discretization: Dict[str, Any] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    """Configuración de modelos."""
    enabled: List[str] = Field(default_factory=lambda: ["hmm_gaussian", "hmm_discrete", "ms_ar"])
    hmm_gaussian: Dict[str, Any] = Field(default_factory=dict)
    hmm_discrete: Dict[str, Any] = Field(default_factory=dict)
    ms_ar: Dict[str, Any] = Field(default_factory=dict)


class SelectionConfig(BaseModel):
    """Configuración de selección de modelos."""
    primary_metric: str = "bic"
    oos_validation: Dict[str, Any] = Field(default_factory=dict)
    walk_forward: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('primary_metric')
    def validate_metric(cls, v):
        valid_metrics = ['aic', 'bic', 'oos_accuracy', 'oos_brier', 'oos_loglik']
        if v not in valid_metrics:
            raise ValueError(f'primary_metric debe ser uno de: {valid_metrics}')
        return v


class StrategyConfig(BaseModel):
    """Configuración de estrategia de trading."""
    regime_mapping: Dict[Union[str, int], str] = Field(default_factory=dict)
    filters: Dict[str, Any] = Field(default_factory=dict)
    sizing: Dict[str, Any] = Field(default_factory=dict)


class RiskConfig(BaseModel):
    """Configuración de gestión de riesgo."""
    stop_loss: Dict[str, Any] = Field(default_factory=dict)
    take_profit: Dict[str, Any] = Field(default_factory=dict)
    trailing_stop: Dict[str, Any] = Field(default_factory=dict)
    costs: Dict[str, float] = Field(default_factory=dict)


class BacktestConfig(BaseModel):
    """Configuración de backtesting."""
    initial_capital: float = 10000.0
    compound_returns: bool = True
    reinvest_profits: bool = True
    metrics: List[str] = Field(default_factory=list)


class PathsConfig(BaseModel):
    """Configuración de rutas."""
    data_dir: str = "data"
    models_dir: str = "data/models"
    signals_dir: str = "data/signals"
    backtests_dir: str = "data/backtests"
    reports_dir: str = "reports"
    logs_dir: str = "logs"


class Config(BaseModel):
    """Configuración principal del sistema."""
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    forecasting: Dict[str, Any] = Field(default_factory=dict)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    risk_management: RiskConfig = Field(default_factory=RiskConfig)
    backtesting: BacktestConfig = Field(default_factory=BacktestConfig)
    reporting: Dict[str, Any] = Field(default_factory=dict)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: Dict[str, Any] = Field(default_factory=dict)
    mlflow: Dict[str, Any] = Field(default_factory=dict)
    api: Dict[str, Any] = Field(default_factory=dict)
    run_config: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        extra = "allow"  # Permitir campos adicionales


def load_yaml_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Carga un archivo YAML.
    
    Args:
        file_path: Ruta al archivo YAML
        
    Returns:
        Diccionario con el contenido del YAML
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        yaml.YAMLError: Si hay error al parsear el YAML
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error al parsear YAML en {file_path}: {e}")


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fusiona dos configuraciones recursivamente.
    
    Args:
        base_config: Configuración base
        override_config: Configuración que sobrescribe la base
        
    Returns:
        Configuración fusionada
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
            
    return merged


def resolve_paths(config: Dict[str, Any], base_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Resuelve las rutas relativas en la configuración.
    
    Args:
        config: Configuración a procesar
        base_dir: Directorio base para rutas relativas
        
    Returns:
        Configuración con rutas resueltas
    """
    if base_dir is None:
        base_dir = Path.cwd()
    
    config = config.copy()
    
    if 'paths' in config:
        paths = config['paths'].copy()
        for key, path in paths.items():
            if isinstance(path, str):
                path_obj = Path(path)
                if not path_obj.is_absolute():
                    paths[key] = str(base_dir / path_obj)
        config['paths'] = paths
    
    return config


def load_environment_variables() -> Dict[str, str]:
    """
    Carga variables de entorno relevantes para APIs.
    
    Returns:
        Diccionario con variables de entorno
    """
    env_vars = {}
    
    # Variables de Binance
    binance_vars = ['BINANCE_API_KEY', 'BINANCE_API_SECRET', 'BINANCE_TESTNET']
    for var in binance_vars:
        if var in os.environ:
            env_vars[var.lower()] = os.environ[var]
    
    # Variables de MLflow
    mlflow_vars = ['MLFLOW_TRACKING_URI', 'MLFLOW_EXPERIMENT_NAME']
    for var in mlflow_vars:
        if var in os.environ:
            env_vars[var.lower()] = os.environ[var]
    
    return env_vars


def load_config(config_path: Union[str, Path], 
                base_config_path: Optional[Union[str, Path]] = None,
                validate: bool = True) -> Config:
    """
    Carga y valida una configuración completa.
    
    Args:
        config_path: Ruta al archivo de configuración principal
        base_config_path: Ruta al archivo de configuración base (opcional)
        validate: Si validar la configuración con Pydantic
        
    Returns:
        Objeto de configuración validado
        
    Raises:
        FileNotFoundError: Si algún archivo no existe
        ValidationError: Si la configuración no es válida
    """
    config_path = Path(config_path)
    
    # Si no se especifica configuración base, usar default.yaml en el mismo directorio
    if base_config_path is None:
        base_config_path = config_path.parent / "default.yaml"
    else:
        base_config_path = Path(base_config_path)
    
    # Cargar configuración base
    base_config = {}
    if base_config_path.exists():
        logger.info(f"Cargando configuración base: {base_config_path}")
        base_config = load_yaml_file(base_config_path)
    
    # Cargar configuración específica
    logger.info(f"Cargando configuración: {config_path}")
    specific_config = load_yaml_file(config_path)
    
    # Fusionar configuraciones
    merged_config = merge_configs(base_config, specific_config)
    
    # Resolver rutas relativas
    merged_config = resolve_paths(merged_config, config_path.parent.parent)
    
    # Cargar variables de entorno
    env_vars = load_environment_variables()
    if env_vars:
        logger.info(f"Variables de entorno cargadas: {list(env_vars.keys())}")
        
        # Agregar API keys a la configuración si existen
        if 'api' not in merged_config:
            merged_config['api'] = {}
        if 'binance' not in merged_config['api']:
            merged_config['api']['binance'] = {}
            
        for key, value in env_vars.items():
            if key.startswith('binance_'):
                merged_config['api']['binance'][key] = value
            elif key.startswith('mlflow_'):
                if 'mlflow' not in merged_config:
                    merged_config['mlflow'] = {}
                merged_config['mlflow'][key] = value
    
    # Validar configuración
    if validate:
        try:
            config = Config(**merged_config)
            logger.info("Configuración validada exitosamente")
            return config
        except Exception as e:
            logger.error(f"Error al validar configuración: {e}")
            raise
    else:
        # Retornar dict como Config sin validación
        return Config.parse_obj(merged_config)


def save_config(config: Union[Config, Dict[str, Any]], 
                output_path: Union[str, Path]) -> None:
    """
    Guarda una configuración en formato YAML.
    
    Args:
        config: Configuración a guardar
        output_path: Ruta donde guardar el archivo
    """
    output_path = Path(output_path)
    
    if isinstance(config, Config):
        config_dict = config.dict()
    else:
        config_dict = config
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    logger.info(f"Configuración guardada en: {output_path}")


def get_default_config() -> Config:
    """
    Retorna una configuración por defecto.
    
    Returns:
        Configuración por defecto
    """
    return Config()


if __name__ == "__main__":
    # Ejemplo de uso
    try:
        config = load_config("config/binance_spot_example.yaml")
        print("Configuración cargada exitosamente")
        print(f"Exchange: {config.data.exchange}")
        print(f"Symbol: {config.data.symbol}")
        print(f"Timeframe: {config.data.timeframe}")
    except Exception as e:
        print(f"Error: {e}")
