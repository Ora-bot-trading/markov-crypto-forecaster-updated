"""
Markov Crypto Forecaster - Sistema de pronóstico con cadenas de Markov para criptomonedas

Un sistema completo para descargar datos de criptomonedas, entrenar modelos basados en 
cadenas de Markov (HMM Gaussianos/Discretos y Markov Switching), generar pronósticos 
probabilísticos y ejecutar estrategias de trading con backtesting riguroso.

Componentes principales:
- data: Descarga y preprocesamiento de datos (CCXT/Binance)
- models: Implementación de modelos de Markov (HMM, MS-AR)
- forecasting: Sistema de pronóstico probabilístico
- strategy: Lógica de trading y gestión de riesgo
- cli: Interfaz de línea de comandos

Author: Senior Quant Engineer
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Senior Quant Engineer"

# Imports principales
from . import config_loader
from . import logging_utils
from . import utils_time

__all__ = [
    "config_loader",
    "logging_utils", 
    "utils_time",
]
