"""
Selector automático de modelos para markov-crypto-forecaster.

Evalúa múltiples modelos de Markov y selecciona el mejor basado en
criterios de información, validación cruzada temporal y métricas out-of-sample.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import json
from datetime import datetime
import warnings
from sklearn.metrics import log_loss, brier_score_loss, mean_squared_error

from .hmm_gaussian import GaussianHMM
from .hmm_discrete import DiscreteHMM
from .ms_ar import MarkovSwitchingAR
from .base import MarkovModelBase
from ..config_loader import Config
from ..logging_utils import get_logger, log_execution_time
from ..utils_time import walk_forward_splits

logger = get_logger(__name__)


class ModelSelector:
    """
    Selector automático de modelos de Markov.
    
    Evalúa diferentes tipos de modelos y configuraciones para encontrar
    el mejor según múltiples criterios:
    - Criterios de información (AIC, BIC, AICc)
    - Validación cruzada temporal
    - Métricas probabilísticas (Brier score, log-loss)
    - Estabilidad de regímenes
    - Capacidad predictiva
    """
    
    def __init__(self, config: Union[Config, dict]):
        """
        Inicializa el selector de modelos.
        
        Args:
            config: Configuración del sistema
        """
        if isinstance(config, dict):
            self.config = config
        else:
            self.config = config.dict()
            
        self.models_config = self.config.get('models', {})
        self.selection_config = self.config.get('selection', {})
        self.paths_config = self.config.get('paths', {})
        
        # Configuración de selección
        self.primary_metric = self.selection_config.get('primary_metric', 'bic')
        self.enabled_models = self.models_config.get('enabled', ['hmm_gaussian', 'hmm_discrete', 'ms_ar'])
        
        # Configuración de validación
        self.oos_validation = self.selection_config.get('oos_validation', {})
        self.walk_forward_config = self.selection_config.get('walk_forward', {})
        
        # Resultados
        self.evaluation_results = {}
        self.best_model = None
        self.best_config = None
        self.selection_report = {}
        
        # Crear directorio de resultados
        self.models_dir = Path(self.paths_config.get('models_dir', 'data/models'))
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Selector inicializado: métrica={self.primary_metric}, modelos={self.enabled_models}")
    
    @log_execution_time("select_best_model")
    def select_best_model(self, X: pd.DataFrame, 
                         y: Optional[pd.Series] = None,
                         target_column: str = 'log_return_1',
                         validation_method: str = 'information_criteria') -> MarkovModelBase:
        """
        Selecciona el mejor modelo evaluando múltiples candidatos.
        
        Args:
            X: Features de entrada
            y: Serie objetivo (opcional)
            target_column: Columna objetivo para MS-AR
            validation_method: 'information_criteria', 'time_series_cv', 'walk_forward'
            
        Returns:
            Mejor modelo entrenado
            
        Raises:
            ValueError: Si no se pueden evaluar modelos
        """
        logger.info(f"Iniciando selección de modelo con {len(X)} observaciones")
        logger.info(f"Método de validación: {validation_method}")
        
        # Generar configuraciones de modelos a evaluar
        model_configs = self._generate_model_configurations()
        logger.info(f"Evaluando {len(model_configs)} configuraciones de modelos")
        
        # Evaluar modelos según el método seleccionado
        if validation_method == 'information_criteria':
            self.evaluation_results = self._evaluate_with_information_criteria(
                X, y, target_column, model_configs
            )
        elif validation_method == 'time_series_cv':
            self.evaluation_results = self._evaluate_with_time_series_cv(
                X, y, target_column, model_configs
            )
        elif validation_method == 'walk_forward':
            self.evaluation_results = self._evaluate_with_walk_forward(
                X, y, target_column, model_configs
            )
        else:
            raise ValueError(f"Método de validación desconocido: {validation_method}")
        
        # Seleccionar mejor modelo
        self.best_config = self._select_best_configuration()
        
        if self.best_config is None:
            raise ValueError("No se pudo seleccionar un modelo válido")
        
        # Entrenar mejor modelo en todo el dataset
        self.best_model = self._train_best_model(X, y, target_column)
        
        # Generar reporte de selección
        self._generate_selection_report()
        
        logger.info(f"Mejor modelo seleccionado: {self.best_config['model_type']} "
                   f"con {self.best_config.get('n_components', self.best_config.get('k_regimes', 'N/A'))} estados")
        
        return self.best_model
    
    def _generate_model_configurations(self) -> List[Dict[str, Any]]:
        """
        Genera todas las configuraciones de modelos a evaluar.
        
        Returns:
            Lista de configuraciones de modelos
        """
        configurations = []
        
        # HMM Gaussiano
        if 'hmm_gaussian' in self.enabled_models:
            hmm_config = self.models_config.get('hmm_gaussian', {})
            n_components_range = hmm_config.get('n_components_range', [2, 3, 4])
            covariance_types = [hmm_config.get('covariance_type', 'full')]
            
            for n_comp in n_components_range:
                for cov_type in covariance_types:
                    config = {
                        'model_type': 'hmm_gaussian',
                        'n_components': n_comp,
                        'covariance_type': cov_type,
                        'model_config': hmm_config
                    }
                    configurations.append(config)
        
        # HMM Discreto
        if 'hmm_discrete' in self.enabled_models:
            hmm_discrete_config = self.models_config.get('hmm_discrete', {})
            n_components_range = hmm_discrete_config.get('n_components_range', [2, 3, 4])
            
            for n_comp in n_components_range:
                config = {
                    'model_type': 'hmm_discrete',
                    'n_components': n_comp,
                    'model_config': hmm_discrete_config
                }
                configurations.append(config)
        
        # Markov Switching AR
        if 'ms_ar' in self.enabled_models:
            ms_ar_config = self.models_config.get('ms_ar', {})
            k_regimes_range = ms_ar_config.get('k_regimes_range', [2, 3])
            order_range = ms_ar_config.get('order_range', [1, 2])
            
            for k_reg in k_regimes_range:
                for order in order_range:
                    config = {
                        'model_type': 'ms_ar',
                        'k_regimes': k_reg,
                        'order': order,
                        'model_config': ms_ar_config
                    }
                    configurations.append(config)
        
        return configurations
    
    def _create_model_from_config(self, config: Dict[str, Any]) -> MarkovModelBase:
        """
        Crea modelo desde configuración.
        
        Args:
            config: Configuración del modelo
            
        Returns:
            Instancia del modelo
        """
        model_type = config['model_type']
        
        if model_type == 'hmm_gaussian':
            return GaussianHMM(
                self.config,
                n_components=config['n_components']
            )
        elif model_type == 'hmm_discrete':
            return DiscreteHMM(
                self.config,
                n_components=config['n_components']
            )
        elif model_type == 'ms_ar':
            return MarkovSwitchingAR(
                self.config,
                k_regimes=config['k_regimes'],
                order=config['order']
            )
        else:
            raise ValueError(f"Tipo de modelo desconocido: {model_type}")
    
    def _evaluate_with_information_criteria(self, X: pd.DataFrame, 
                                          y: Optional[pd.Series],
                                          target_column: str,
                                          model_configs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Evalúa modelos usando criterios de información.
        
        Args:
            X: Features
            y: Serie objetivo
            target_column: Columna objetivo
            model_configs: Configuraciones a evaluar
            
        Returns:
            Dict con resultados de evaluación
        """
        results = {}
        
        for i, config in enumerate(model_configs):
            config_id = f"{config['model_type']}_{i}"
            logger.info(f"Evaluando configuración {i+1}/{len(model_configs)}: {config_id}")
            
            try:
                # Crear y entrenar modelo
                model = self._create_model_from_config(config)
                
                if config['model_type'] == 'ms_ar':
                    model.fit(X, y, target_column=target_column)
                else:
                    model.fit(X, y)
                
                # Calcular criterios de información
                info_criteria = model.calculate_information_criteria(X)
                
                # Métricas adicionales
                health_check = model.validate_model_health()
                training_metrics = model.training_metrics
                
                # Calcular score compuesto
                composite_score = self._calculate_composite_score(info_criteria, training_metrics)
                
                results[config_id] = {
                    'config': config,
                    'model': model,
                    'info_criteria': info_criteria,
                    'training_metrics': training_metrics,
                    'health_check': health_check,
                    'composite_score': composite_score,
                    'evaluation_method': 'information_criteria'
                }
                
                logger.info(f"  {config_id}: {self.primary_metric.upper()}={info_criteria[self.primary_metric]:.2f}")
                
            except Exception as e:
                logger.error(f"Error evaluando {config_id}: {e}")
                results[config_id] = {
                    'config': config,
                    'model': None,
                    'error': str(e),
                    'evaluation_method': 'information_criteria'
                }
        
        return results
    
    def _evaluate_with_time_series_cv(self, X: pd.DataFrame, 
                                    y: Optional[pd.Series],
                                    target_column: str,
                                    model_configs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Evalúa modelos usando validación cruzada temporal.
        
        Args:
            X: Features
            y: Serie objetivo
            target_column: Columna objetivo
            model_configs: Configuraciones a evaluar
            
        Returns:
            Dict con resultados de evaluación
        """
        cv_config = self.oos_validation
        n_splits = cv_config.get('n_splits', 5)
        test_size = cv_config.get('test_size', 0.2)
        
        results = {}
        
        # Crear splits temporales
        split_size = int(len(X) * test_size)
        step_size = len(X) // (n_splits + 1)
        
        for i, config in enumerate(model_configs):
            config_id = f"{config['model_type']}_{i}"
            logger.info(f"Evaluando {config_id} con CV temporal")
            
            cv_scores = []
            cv_metrics = []
            
            try:
                for fold in range(n_splits):
                    # Crear split temporal
                    test_start = (fold + 1) * step_size
                    test_end = test_start + split_size
                    
                    if test_end > len(X):
                        break
                    
                    train_data = X.iloc[:test_start]
                    test_data = X.iloc[test_start:test_end]
                    
                    if len(train_data) < 50:  # Mínimo para entrenar
                        continue
                    
                    # Entrenar modelo
                    model = self._create_model_from_config(config)
                    
                    if config['model_type'] == 'ms_ar':
                        train_y = train_data[target_column] if y is None else y.iloc[:test_start]
                        model.fit(train_data, train_y, target_column=target_column)
                    else:
                        train_y = y.iloc[:test_start] if y is not None else None
                        model.fit(train_data, train_y)
                    
                    # Evaluar en test
                    test_score = model.score(test_data)
                    cv_scores.append(test_score)
                    
                    # Métricas adicionales
                    test_states = model.predict_states(test_data)
                    test_probs = model.predict_state_probabilities(test_data)
                    
                    fold_metrics = {
                        'log_likelihood': test_score,
                        'n_test_samples': len(test_data),
                        'state_diversity': len(np.unique(test_states)) / model.n_states
                    }
                    cv_metrics.append(fold_metrics)
                
                if cv_scores:
                    # Agregar resultados de CV
                    cv_mean = np.mean(cv_scores)
                    cv_std = np.std(cv_scores)
                    
                    results[config_id] = {
                        'config': config,
                        'cv_scores': cv_scores,
                        'cv_mean': cv_mean,
                        'cv_std': cv_std,
                        'cv_metrics': cv_metrics,
                        'n_folds_completed': len(cv_scores),
                        'evaluation_method': 'time_series_cv'
                    }
                    
                    logger.info(f"  {config_id}: CV Score={cv_mean:.2f} ± {cv_std:.2f}")
                else:
                    logger.warning(f"No se completaron folds para {config_id}")
                    
            except Exception as e:
                logger.error(f"Error en CV para {config_id}: {e}")
                results[config_id] = {
                    'config': config,
                    'error': str(e),
                    'evaluation_method': 'time_series_cv'
                }
        
        return results
    
    def _evaluate_with_walk_forward(self, X: pd.DataFrame, 
                                  y: Optional[pd.Series],
                                  target_column: str,
                                  model_configs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Evalúa modelos usando walk-forward validation.
        
        Args:
            X: Features
            y: Serie objetivo
            target_column: Columna objetivo
            model_configs: Configuraciones a evaluar
            
        Returns:
            Dict con resultados de evaluación
        """
        wf_config = self.walk_forward_config
        train_window = wf_config.get('train_window', 120)
        test_window = wf_config.get('test_window', 30)
        method = wf_config.get('method', 'rolling')
        
        results = {}
        
        # Crear splits de walk-forward
        from ..utils_time import walk_forward_splits
        try:
            splits = walk_forward_splits(X, train_window, test_window, method=method)
            logger.info(f"Creados {len(splits)} splits para walk-forward")
        except Exception as e:
            logger.error(f"Error creando splits: {e}")
            return {}
        
        for i, config in enumerate(model_configs):
            config_id = f"{config['model_type']}_{i}"
            logger.info(f"Evaluando {config_id} con walk-forward")
            
            split_results = []
            
            try:
                for split_idx, (train_df, test_df) in enumerate(splits):
                    if len(train_df) < 50 or len(test_df) < 5:
                        continue
                    
                    # Entrenar modelo
                    model = self._create_model_from_config(config)
                    
                    if config['model_type'] == 'ms_ar':
                        model.fit(train_df, target_column=target_column)
                    else:
                        model.fit(train_df)
                    
                    # Evaluar
                    test_score = model.score(test_df)
                    test_states = model.predict_states(test_df)
                    
                    split_result = {
                        'split_idx': split_idx,
                        'train_start': train_df.index.min(),
                        'train_end': train_df.index.max(),
                        'test_start': test_df.index.min(),
                        'test_end': test_df.index.max(),
                        'test_score': test_score,
                        'n_train': len(train_df),
                        'n_test': len(test_df),
                        'n_states_used': len(np.unique(test_states))
                    }
                    split_results.append(split_result)
                
                if split_results:
                    # Agregar estadísticas
                    test_scores = [r['test_score'] for r in split_results]
                    wf_mean = np.mean(test_scores)
                    wf_std = np.std(test_scores)
                    wf_median = np.median(test_scores)
                    
                    results[config_id] = {
                        'config': config,
                        'split_results': split_results,
                        'wf_mean': wf_mean,
                        'wf_std': wf_std,
                        'wf_median': wf_median,
                        'n_splits_completed': len(split_results),
                        'evaluation_method': 'walk_forward'
                    }
                    
                    logger.info(f"  {config_id}: WF Score={wf_mean:.2f} ± {wf_std:.2f}")
                
            except Exception as e:
                logger.error(f"Error en walk-forward para {config_id}: {e}")
                results[config_id] = {
                    'config': config,
                    'error': str(e),
                    'evaluation_method': 'walk_forward'
                }
        
        return results
    
    def _calculate_composite_score(self, info_criteria: Dict[str, float], 
                                 training_metrics: Dict[str, Any]) -> float:
        """
        Calcula score compuesto para ranking de modelos.
        
        Args:
            info_criteria: Criterios de información
            training_metrics: Métricas de entrenamiento
            
        Returns:
            Score compuesto (menor es mejor)
        """
        # Score base usando métrica primaria
        base_score = info_criteria.get(self.primary_metric, np.inf)
        
        # Penalizaciones
        penalties = 0
        
        # Penalizar por desequilibrio de estados
        state_frequencies = training_metrics.get('state_frequencies', [])
        if state_frequencies:
            min_freq = min(state_frequencies)
            if min_freq < 0.05:  # Estados poco frecuentes
                penalties += 50
        
        # Penalizar por problemas de salud del modelo
        # (esto se evaluaría en _select_best_configuration)
        
        return base_score + penalties
    
    def _select_best_configuration(self) -> Optional[Dict[str, Any]]:
        """
        Selecciona la mejor configuración de los resultados.
        
        Returns:
            Mejor configuración o None si no hay válidas
        """
        valid_results = {
            k: v for k, v in self.evaluation_results.items() 
            if 'error' not in v and v.get('model') is not None
        }
        
        if not valid_results:
            logger.error("No hay resultados válidos para selección")
            return None
        
        # Ordenar según método de evaluación
        evaluation_method = list(valid_results.values())[0]['evaluation_method']
        
        if evaluation_method == 'information_criteria':
            # Usar métrica primaria (menor es mejor para AIC/BIC)
            best_key = min(valid_results.keys(), 
                          key=lambda k: valid_results[k]['info_criteria'][self.primary_metric])
        
        elif evaluation_method == 'time_series_cv':
            # Usar CV mean (mayor es mejor para log-likelihood)
            best_key = max(valid_results.keys(), 
                          key=lambda k: valid_results[k]['cv_mean'])
        
        elif evaluation_method == 'walk_forward':
            # Usar WF mean (mayor es mejor para log-likelihood)
            best_key = max(valid_results.keys(), 
                          key=lambda k: valid_results[k]['wf_mean'])
        else:
            best_key = list(valid_results.keys())[0]
        
        best_result = valid_results[best_key]
        
        # Verificar salud del modelo si está disponible
        if 'health_check' in best_result:
            health = best_result['health_check']
            if not health.get('healthy', True):
                logger.warning(f"Mejor modelo tiene problemas de salud: {health.get('errors', [])}")
        
        logger.info(f"Mejor configuración seleccionada: {best_key}")
        
        return {
            'config_id': best_key,
            'config': best_result['config'],
            'evaluation_result': best_result
        }
    
    def _train_best_model(self, X: pd.DataFrame, 
                         y: Optional[pd.Series],
                         target_column: str) -> MarkovModelBase:
        """
        Entrena el mejor modelo en todo el dataset.
        
        Args:
            X: Features completas
            y: Serie objetivo
            target_column: Columna objetivo
            
        Returns:
            Modelo entrenado
        """
        config = self.best_config['config']
        logger.info(f"Entrenando mejor modelo en dataset completo: {config['model_type']}")
        
        model = self._create_model_from_config(config)
        
        if config['model_type'] == 'ms_ar':
            model.fit(X, y, target_column=target_column)
        else:
            model.fit(X, y)
        
        return model
    
    def _generate_selection_report(self):
        """Genera reporte de selección de modelo."""
        self.selection_report = {
            'selection_timestamp': datetime.now().isoformat(),
            'primary_metric': self.primary_metric,
            'evaluation_method': self.best_config['evaluation_result']['evaluation_method'],
            'best_model': {
                'config_id': self.best_config['config_id'],
                'model_type': self.best_config['config']['model_type'],
                'n_states': self.best_config['config'].get('n_components', 
                                                         self.best_config['config'].get('k_regimes')),
                'specific_config': self.best_config['config']
            },
            'all_results_summary': {}
        }
        
        # Resumen de todos los resultados
        for config_id, result in self.evaluation_results.items():
            if 'error' not in result:
                if result['evaluation_method'] == 'information_criteria':
                    summary = {
                        'model_type': result['config']['model_type'],
                        'primary_score': result['info_criteria'][self.primary_metric],
                        'aic': result['info_criteria']['aic'],
                        'bic': result['info_criteria']['bic']
                    }
                elif result['evaluation_method'] == 'time_series_cv':
                    summary = {
                        'model_type': result['config']['model_type'],
                        'cv_mean': result['cv_mean'],
                        'cv_std': result['cv_std'],
                        'n_folds': result['n_folds_completed']
                    }
                elif result['evaluation_method'] == 'walk_forward':
                    summary = {
                        'model_type': result['config']['model_type'],
                        'wf_mean': result['wf_mean'],
                        'wf_std': result['wf_std'],
                        'n_splits': result['n_splits_completed']
                    }
                else:
                    summary = {'model_type': result['config']['model_type']}
                
                self.selection_report['all_results_summary'][config_id] = summary
            else:
                self.selection_report['all_results_summary'][config_id] = {
                    'model_type': result['config']['model_type'],
                    'error': result['error']
                }
    
    def save_selection_report(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        """
        Guarda reporte de selección.
        
        Args:
            filepath: Ruta donde guardar (opcional)
            
        Returns:
            Path del archivo guardado
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_selection_report_{timestamp}.json"
            filepath = self.models_dir / filename
        else:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convertir numpy types para JSON
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            return obj
        
        # Preparar reporte para JSON
        json_report = json.loads(json.dumps(self.selection_report, default=convert_numpy))
        
        with open(filepath, 'w') as f:
            json.dump(json_report, f, indent=2)
        
        logger.info(f"Reporte de selección guardado: {filepath}")
        return filepath
    
    def get_ranking(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Obtiene ranking de modelos evaluados.
        
        Args:
            top_k: Número de mejores modelos a retornar
            
        Returns:
            Lista de modelos ordenados por performance
        """
        valid_results = {
            k: v for k, v in self.evaluation_results.items() 
            if 'error' not in v
        }
        
        if not valid_results:
            return []
        
        evaluation_method = list(valid_results.values())[0]['evaluation_method']
        
        # Ordenar según método
        if evaluation_method == 'information_criteria':
            sorted_items = sorted(
                valid_results.items(),
                key=lambda x: x[1]['info_criteria'][self.primary_metric]
            )
        elif evaluation_method == 'time_series_cv':
            sorted_items = sorted(
                valid_results.items(),
                key=lambda x: x[1]['cv_mean'],
                reverse=True
            )
        elif evaluation_method == 'walk_forward':
            sorted_items = sorted(
                valid_results.items(),
                key=lambda x: x[1]['wf_mean'],
                reverse=True
            )
        else:
            sorted_items = list(valid_results.items())
        
        # Crear ranking
        ranking = []
        for i, (config_id, result) in enumerate(sorted_items[:top_k]):
            rank_info = {
                'rank': i + 1,
                'config_id': config_id,
                'model_type': result['config']['model_type'],
                'n_states': result['config'].get('n_components', 
                                                result['config'].get('k_regimes')),
                'evaluation_method': evaluation_method
            }
            
            if evaluation_method == 'information_criteria':
                rank_info.update({
                    'primary_score': result['info_criteria'][self.primary_metric],
                    'aic': result['info_criteria']['aic'],
                    'bic': result['info_criteria']['bic']
                })
            elif evaluation_method == 'time_series_cv':
                rank_info.update({
                    'cv_mean': result['cv_mean'],
                    'cv_std': result['cv_std']
                })
            elif evaluation_method == 'walk_forward':
                rank_info.update({
                    'wf_mean': result['wf_mean'],
                    'wf_std': result['wf_std']
                })
            
            ranking.append(rank_info)
        
        return ranking


if __name__ == "__main__":
    # Ejemplo de uso
    try:
        from ..config_loader import load_config
        
        config = load_config("config/binance_spot_example.yaml")
        
        # Crear datos de ejemplo con múltiples features
        np.random.seed(42)
        n_samples = 200
        
        # Simular regímenes
        regime_changes = [0, 70, 140, 200]
        data = []
        
        for i, (start, end) in enumerate(zip(regime_changes[:-1], regime_changes[1:])):
            regime_size = end - start
            
            if i == 0:  # Régimen bajista
                returns = np.random.normal(-0.01, 0.02, regime_size)
                volatility = np.random.normal(0.03, 0.01, regime_size)
                volume = np.random.normal(0.8, 0.2, regime_size)
            elif i == 1:  # Régimen alcista
                returns = np.random.normal(0.015, 0.015, regime_size)
                volatility = np.random.normal(0.02, 0.005, regime_size)
                volume = np.random.normal(1.2, 0.3, regime_size)
            else:  # Régimen lateral
                returns = np.random.normal(0.001, 0.01, regime_size)
                volatility = np.random.normal(0.015, 0.003, regime_size)
                volume = np.random.normal(1.0, 0.15, regime_size)
            
            regime_data = np.column_stack([returns, volatility, volume])
            data.append(regime_data)
        
        X = pd.DataFrame(
            np.vstack(data),
            columns=['log_return_1', 'volatility', 'volume_zscore']
        )
        
        print(f"Datos de ejemplo: {len(X)} muestras, {len(X.columns)} features")
        
        # Seleccionar mejor modelo
        selector = ModelSelector(config)
        best_model = selector.select_best_model(X, validation_method='information_criteria')
        
        print(f"\nMejor modelo seleccionado:")
        print(f"  Tipo: {best_model.model_type}")
        print(f"  Estados: {best_model.n_states}")
        print(f"  Features: {len(best_model.feature_names)}")
        
        # Mostrar ranking
        ranking = selector.get_ranking(top_k=3)
        print(f"\nRanking de modelos:")
        for rank_info in ranking:
            print(f"  {rank_info['rank']}. {rank_info['model_type']} "
                  f"({rank_info['n_states']} estados): "
                  f"{selector.primary_metric.upper()}={rank_info.get('primary_score', 'N/A')}")
        
        # Guardar reporte
        report_path = selector.save_selection_report()
        print(f"\nReporte guardado: {report_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
