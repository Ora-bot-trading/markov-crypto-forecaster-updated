"""
Módulo de modelos para markov-crypto-forecaster.

Implementa modelos basados en cadenas de Markov:
- base: Interfaz común y utilidades base
- hmm_gaussian: Hidden Markov Models Gaussianos
- hmm_discrete: Hidden Markov Models discretos
- ms_ar: Markov Switching Autoregressive models
- selector: Selección automática de modelos
"""

from . import base
from . import hmm_gaussian
from . import hmm_discrete
from . import ms_ar
from . import selector

__all__ = [
    "base",
    "hmm_gaussian",
    "hmm_discrete", 
    "ms_ar",
    "selector"
]
