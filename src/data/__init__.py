"""
Módulo de manejo de datos para markov-crypto-forecaster.

Componentes:
- downloader: Descarga de datos de exchanges (CCXT/Binance)
- preprocessor: Limpieza y preprocesamiento de datos
- feature_engineer: Creación de features para modelos
"""

from . import downloader
from . import preprocessor
from . import feature_engineer

__all__ = [
    "downloader",
    "preprocessor", 
    "feature_engineer"
]
