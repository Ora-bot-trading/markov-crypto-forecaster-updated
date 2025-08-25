"""
Módulo de estrategia de trading para markov-crypto-forecaster.

Componentes:
- position_logic: Lógica de posiciones y señales de trading
- risk: Gestión de riesgo y cálculo de position sizing
- backtester: Sistema de backtesting con walk-forward
- metrics: Cálculo de métricas de performance
- reports: Generación de reportes y visualizaciones
"""

from . import position_logic
from . import risk
from . import backtester
from . import metrics
from . import reports

__all__ = [
    "position_logic",
    "risk",
    "backtester", 
    "metrics",
    "reports"
]
