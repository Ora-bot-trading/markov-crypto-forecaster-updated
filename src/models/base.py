"""
Clase base para modelos de Markov en markov-crypto-forecaster.

Define interfaz común, persistencia, validación y utilidades
compartidas para todos los modelos de cadenas de Markov.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List
import pickle
import json
from datetime import datetime
import warnings

from ..config_loader import Config
from ..logging_utils import get_logger, log_execution_time

logger = get_logger(__name__)


class MarkovModelBase(ABC):
    """
    Clase base abstracta para modelos de Markov.
    
    Define la interfaz común que deben implementar todos los modelos:
    HMM Gaussiano, HMM Discreto, Markov Switching AR, etc.
    """
    
    def __init__(self, config: Union[Config, dict], model_type: str):
        """
        Inicializa el modelo base.
        
        Args:
            config: Configuración del sistema
            model_type: Tipo de modelo ('hmm_gaussian', 'hmm_discrete', 'ms_ar')
        """
        if isinstance(config, dict):
            self.config = config
        else:
            self.config = config.dict()
            
        self.model_type = model_type
        self.models_config = self.config.get('models', {})
        self.model_config = self.models_config.get(model_type, {})
        self.paths_config = self.config.get('paths', {})
        
        # Estado del modelo
        self.is_fitted = False
        self.model = None
        self.feature_names = None
        self.n_states = None
        self.random_state = self.model_config.get('random_state', 42)
        
        # Métricas y resultados
        self.training_metrics = {}
        self.validation_metrics = {}
        self.model_info = {}
        
        # Configurar random seed
        np.random.seed(self.random_state)
        
        # Crear directorio de modelos
        self.models_dir = Path(self.paths_config.get('models_dir', 'data/models'))
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs) -> 'MarkovModelBase':
        """
        Entrena el modelo con los datos.
        
        Args:
            X: Features de entrada
            y: Variable objetivo (opcional, depende del modelo)
            **kwargs: Argumentos adicionales específicos del modelo
            
        Returns:
            Self para method chaining
        """
        pass
    
    @abstractmethod
    def predict_states(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predice la secuencia de estados más probable (Viterbi).
        
        Args:
            X: Features de entrada
            
        Returns:
            Array con secuencia de estados
        """
        pass
    
    @abstractmethod
    def predict_state_probabilities(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calcula probabilidades de estado para cada observación.
        
        Args:
            X: Features de entrada
            
        Returns:
            Array con probabilidades [n_samples, n_states]
        """
        pass
    
    @abstractmethod
    def sample(self, n_samples: int, random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genera muestras del modelo.
        
        Args:
            n_samples: Número de muestras a generar
            random_state: Semilla aleatoria
            
        Returns:
            Tupla (observaciones, estados)
        """
        pass
    
    @abstractmethod
    def score(self, X: pd.DataFrame) -> float:
        """
        Calcula log-likelihood de los datos.
        
        Args:
            X: Features de entrada
            
        Returns:
            Log-likelihood
        """
        pass
    
    def validate_input(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Valida y prepara datos de entrada.
        
        Args:
            X: Features de entrada
            y: Variable objetivo (opcional)
            
        Returns:
            Tupla (X_array, y_array) como numpy arrays
            
        Raises:
            ValueError: Si los datos no son válidos
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X debe ser un pandas DataFrame")
        
        if len(X) == 0:
            raise ValueError("X no puede estar vacío")
        
        # Verificar features si el modelo ya fue entrenado
        if self.is_fitted and self.feature_names is not None:
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                raise ValueError(f"Features faltantes: {missing_features}")
            
            # Reordenar columnas para mantener consistencia
            X = X[self.feature_names]
        
        # Convertir a numpy y verificar NaN
        X_array = X.values
        if np.isnan(X_array).any():
            # Log warning pero no fallar - algunos modelos pueden manejar NaN
            nan_count = np.isnan(X_array).sum()
            total_count = X_array.size
            logger.warning(f"Detectados {nan_count}/{total_count} valores NaN en X")
        
        # Validar y
        y_array = None
        if y is not None:
            if not isinstance(y, (pd.Series, np.ndarray)):
                raise ValueError("y debe ser pandas Series o numpy array")
            
            if len(y) != len(X):
                raise ValueError(f"Longitudes no coinciden: X={len(X)}, y={len(y)}")
            
            y_array = y.values if isinstance(y, pd.Series) else y
            
            if np.isnan(y_array).any():
                nan_count = np.isnan(y_array).sum()
                logger.warning(f"Detectados {nan_count} valores NaN en y")
        
        return X_array, y_array
    
    def calculate_information_criteria(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        Calcula criterios de información (AIC, BIC).
        
        Args:
            X: Features de entrada
            
        Returns:
            Dict con criterios de información
        """
        if not self.is_fitted:
            raise ValueError("Modelo debe estar entrenado")
        
        n_samples = len(X)
        log_likelihood = self.score(X)
        
        # Número de parámetros - implementación base, los submodelos pueden sobrescribir
        n_params = self.get_n_parameters()
        
        # AIC = -2 * log_likelihood + 2 * n_params
        aic = -2 * log_likelihood + 2 * n_params
        
        # BIC = -2 * log_likelihood + log(n_samples) * n_params
        bic = -2 * log_likelihood + np.log(n_samples) * n_params
        
        # AICc (AIC corregido para muestras pequeñas)
        if n_samples > n_params + 1:
            aicc = aic + (2 * n_params * (n_params + 1)) / (n_samples - n_params - 1)
        else:
            aicc = np.inf
        
        return {
            'log_likelihood': log_likelihood,
            'aic': aic,
            'bic': bic,
            'aicc': aicc,
            'n_parameters': n_params,
            'n_samples': n_samples
        }
    
    @abstractmethod
    def get_n_parameters(self) -> int:
        """
        Retorna el número de parámetros del modelo.
        
        Returns:
            Número de parámetros
        """
        pass
    
    def get_transition_matrix(self) -> Optional[np.ndarray]:
        """
        Obtiene la matriz de transición del modelo.
        
        Returns:
            Matriz de transición [n_states, n_states] o None si no aplica
        """
        # Implementación por defecto - los submodelos pueden sobrescribir
        return None
    
    def get_emission_parameters(self) -> Optional[Dict[str, Any]]:
        """
        Obtiene parámetros de emisión del modelo.
        
        Returns:
            Dict con parámetros de emisión o None si no aplica
        """
        # Implementación por defecto - los submodelos pueden sobrescribir
        return None
    
    @log_execution_time("save_model")
    def save_model(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        """
        Guarda el modelo entrenado.
        
        Args:
            filepath: Ruta donde guardar (opcional)
            
        Returns:
            Path del archivo guardado
            
        Raises:
            ValueError: Si el modelo no está entrenado
        """
        if not self.is_fitted:
            raise ValueError("No se puede guardar un modelo no entrenado")
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.model_type}_{self.n_states}states_{timestamp}.pkl"
            filepath = self.models_dir / filename
        else:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Preparar datos para guardar
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'model_config': self.model_config,
            'feature_names': self.feature_names,
            'n_states': self.n_states,
            'random_state': self.random_state,
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics,
            'model_info': self.model_info,
            'timestamp': datetime.now().isoformat()
        }
        
        # Guardar con pickle
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Guardar metadata en JSON para inspección
        metadata_path = filepath.with_suffix('.json')
        metadata = {k: v for k, v in model_data.items() if k != 'model'}  # Excluir el objeto modelo
        
        # Serializar numpy arrays para JSON
        for key, value in metadata.items():
            if isinstance(value, np.ndarray):
                metadata[key] = value.tolist()
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, np.ndarray):
                        metadata[key][subkey] = subvalue.tolist()
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Modelo guardado en: {filepath}")
        return filepath
    
    @classmethod
    def load_model(cls, filepath: Union[str, Path], config: Optional[Union[Config, dict]] = None) -> 'MarkovModelBase':
        """
        Carga un modelo guardado.
        
        Args:
            filepath: Ruta del archivo del modelo
            config: Configuración (opcional, se usa la guardada si no se proporciona)
            
        Returns:
            Instancia del modelo cargado
            
        Raises:
            FileNotFoundError: Si el archivo no existe
            ValueError: Si el modelo no es válido
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Archivo de modelo no encontrado: {filepath}")
        
        # Cargar datos
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Verificar datos del modelo
        required_keys = ['model', 'model_type', 'feature_names', 'n_states']
        missing_keys = [key for key in required_keys if key not in model_data]
        if missing_keys:
            raise ValueError(f"Datos del modelo incompletos. Faltantes: {missing_keys}")
        
        # Usar configuración guardada si no se proporciona
        if config is None:
            config = model_data.get('model_config', {})
        
        # Crear instancia según el tipo
        model_type = model_data['model_type']
        
        if model_type == 'hmm_gaussian':
            from .hmm_gaussian import GaussianHMM
            instance = GaussianHMM(config)
        elif model_type == 'hmm_discrete':
            from .hmm_discrete import DiscreteHMM
            instance = DiscreteHMM(config)
        elif model_type == 'ms_ar':
            from .ms_ar import MarkovSwitchingAR
            instance = MarkovSwitchingAR(config)
        else:
            raise ValueError(f"Tipo de modelo desconocido: {model_type}")
        
        # Restaurar estado
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.n_states = model_data['n_states']
        instance.random_state = model_data.get('random_state', 42)
        instance.training_metrics = model_data.get('training_metrics', {})
        instance.validation_metrics = model_data.get('validation_metrics', {})
        instance.model_info = model_data.get('model_info', {})
        instance.is_fitted = True
        
        logger.info(f"Modelo cargado desde: {filepath}")
        return instance
    
    def validate_model_health(self) -> Dict[str, Any]:
        """
        Valida la salud del modelo entrenado.
        
        Returns:
            Dict con métricas de salud del modelo
        """
        if not self.is_fitted:
            return {'healthy': False, 'reason': 'Modelo no entrenado'}
        
        health_report = {
            'healthy': True,
            'checks': {},
            'warnings': [],
            'errors': []
        }
        
        try:
            # 1. Verificar matriz de transición
            transition_matrix = self.get_transition_matrix()
            if transition_matrix is not None:
                # Verificar que las filas suman a 1
                row_sums = np.sum(transition_matrix, axis=1)
                if not np.allclose(row_sums, 1.0, rtol=1e-3):
                    health_report['errors'].append("Matriz de transición no está normalizada")
                    health_report['healthy'] = False
                
                # Verificar que no hay valores negativos
                if np.any(transition_matrix < 0):
                    health_report['errors'].append("Matriz de transición tiene valores negativos")
                    health_report['healthy'] = False
                
                health_report['checks']['transition_matrix'] = 'ok' if health_report['healthy'] else 'error'
            
            # 2. Verificar convergencia (si está disponible)
            if hasattr(self.model, 'monitor_'):
                convergence = self.model.monitor_
                if hasattr(convergence, 'converged') and not convergence.converged:
                    health_report['warnings'].append("Modelo no convergió completamente")
                health_report['checks']['convergence'] = 'ok' if convergence.converged else 'warning'
            
            # 3. Verificar número de estados
            if self.n_states is None or self.n_states <= 0:
                health_report['errors'].append("Número de estados inválido")
                health_report['healthy'] = False
            
            health_report['checks']['n_states'] = 'ok' if self.n_states and self.n_states > 0 else 'error'
            
            # 4. Verificar features
            if self.feature_names is None or len(self.feature_names) == 0:
                health_report['errors'].append("No hay features definidas")
                health_report['healthy'] = False
            
            health_report['checks']['features'] = 'ok' if self.feature_names else 'error'
            
        except Exception as e:
            health_report['errors'].append(f"Error en validación: {str(e)}")
            health_report['healthy'] = False
        
        return health_report
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Obtiene resumen completo del modelo.
        
        Returns:
            Dict con información del modelo
        """
        summary = {
            'model_type': self.model_type,
            'n_states': self.n_states,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
            'random_state': self.random_state,
            'config': self.model_config
        }
        
        if self.is_fitted:
            summary.update({
                'training_metrics': self.training_metrics,
                'validation_metrics': self.validation_metrics,
                'model_info': self.model_info,
                'health_check': self.validate_model_health()
            })
            
            # Agregar información específica del modelo
            transition_matrix = self.get_transition_matrix()
            if transition_matrix is not None:
                summary['transition_matrix'] = transition_matrix.tolist()
            
            emission_params = self.get_emission_parameters()
            if emission_params is not None:
                summary['emission_parameters'] = emission_params
        
        return summary
    
    def __repr__(self) -> str:
        """Representación string del modelo."""
        status = "fitted" if self.is_fitted else "not fitted"
        states = f"{self.n_states} states" if self.n_states else "unknown states"
        features = f"{len(self.feature_names)} features" if self.feature_names else "unknown features"
        
        return f"{self.__class__.__name__}({states}, {features}, {status})"


class ModelValidationMixin:
    """
    Mixin para validación común de modelos.
    
    Proporciona métodos de validación que pueden ser utilizados
    por cualquier modelo de Markov.
    """
    
    @staticmethod
    def validate_n_states(n_states: int, max_states: int = 20) -> int:
        """
        Valida el número de estados.
        
        Args:
            n_states: Número de estados propuesto
            max_states: Máximo número de estados permitido
            
        Returns:
            Número de estados validado
            
        Raises:
            ValueError: Si n_states no es válido
        """
        if not isinstance(n_states, int) or n_states <= 0:
            raise ValueError("n_states debe ser un entero positivo")
        
        if n_states > max_states:
            warnings.warn(f"Número de estados {n_states} > {max_states} puede causar overfitting")
        
        return n_states
    
    @staticmethod
    def validate_features_for_states(n_features: int, n_states: int) -> bool:
        """
        Valida que hay suficientes features para el número de estados.
        
        Args:
            n_features: Número de features
            n_states: Número de estados
            
        Returns:
            True si la relación es válida
            
        Raises:
            ValueError: Si no hay suficientes features
        """
        min_features = n_states // 2 + 1  # Regla heurística
        
        if n_features < min_features:
            raise ValueError(
                f"Muy pocas features ({n_features}) para {n_states} estados. "
                f"Mínimo recomendado: {min_features}"
            )
        
        return True
    
    @staticmethod
    def validate_sample_size(n_samples: int, n_states: int, n_features: int) -> bool:
        """
        Valida que hay suficientes muestras para entrenar.
        
        Args:
            n_samples: Número de muestras
            n_states: Número de estados
            n_features: Número de features
            
        Returns:
            True si el tamaño es adecuado
            
        Raises:
            ValueError: Si hay muy pocas muestras
        """
        # Regla heurística: al menos 10 observaciones por parámetro
        min_samples = n_states * (n_states + n_features) * 10
        
        if n_samples < min_samples:
            warnings.warn(
                f"Pocas muestras ({n_samples}) para modelo con {n_states} estados "
                f"y {n_features} features. Recomendado: ≥{min_samples}"
            )
        
        # Mínimo absoluto
        absolute_min = n_states * 20
        if n_samples < absolute_min:
            raise ValueError(
                f"Muy pocas muestras ({n_samples}). "
                f"Mínimo absoluto: {absolute_min}"
            )
        
        return True


if __name__ == "__main__":
    # La clase base no se puede instanciar directamente
    print("MarkovModelBase es una clase abstracta.")
    print("Usar HMM Gaussiano, HMM Discreto o Markov Switching AR.")
