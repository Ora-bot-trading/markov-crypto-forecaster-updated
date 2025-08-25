# Markov Crypto Forecaster

Un sistema completo de trading cuantitativo basado en cadenas de Markov para criptomonedas, desarrollado por un Senior Quant Engineer con experiencia en series temporales financieras.

## 🚀 Características Principales

- **Modelos de Markov Avanzados**: HMM Gaussianos, HMM Discretos y Markov Switching AR
- **Sin Look-Ahead Bias**: Pipeline de datos y features diseñado para evitar filtraciones futuras
- **Backtesting Riguroso**: Walk-forward analysis con costos y slippage realistas
- **Gestión de Riesgo**: Stop loss adaptativos, take profit y trailing stops basados en ATR
- **API Integration**: Soporte para Binance y otros exchanges via CCXT
- **CLI Completa**: Interfaz de línea de comandos para todas las operaciones
- **Escalable**: Diseño modular listo para producción

## 📊 Modelos Implementados

### Hidden Markov Models (HMM)
- **HMM Gaussiano**: Emisiones multivariantes con diferentes tipos de covarianza
- **HMM Discreto**: Features cuantizadas para modelado categórico

### Markov Switching Models
- **MS-AR**: Modelos autorregresivos con cambio de régimen
- **Switching Variance**: Varianza que cambia según el régimen

### Selección Automática
- Criterios de información (AIC, BIC, AICc)
- Validación cruzada temporal
- Walk-forward validation

## 🛠️ Instalación

### Prerrequisitos
- Python 3.11+
- TA-Lib (para indicadores técnicos)

### Instalación rápida

```bash
# Clonar repositorio
git clone <repo-url>
cd markov-crypto-forecaster

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar TA-Lib (necesario para indicadores técnicos)
# En macOS con Homebrew:
brew install ta-lib

# En Ubuntu:
sudo apt-get install libta-lib-dev

# En Windows: descargar binarios desde https://mrjbq7.github.io/ta-lib/install.html
```

### Configuración de APIs

Crear archivo `.env` en el directorio raíz:

```env
# Binance API (opcional para live trading)
BINANCE_API_KEY=tu_api_key_aqui
BINANCE_API_SECRET=tu_api_secret_aqui
BINANCE_TESTNET=true

# MLflow (opcional)
MLFLOW_TRACKING_URI=./mlruns
MLFLOW_EXPERIMENT_NAME=markov_crypto_forecaster
```

## 🚀 Uso Rápido

### 1. Descargar Datos

```bash
python -m src.cli.main download-data --config config/binance_spot_example.yaml
```

### 2. Crear Features

```bash
python -m src.cli.main make-features --config config/binance_spot_example.yaml
```

### 3. Entrenar Modelos

```bash
python -m src.cli.main train --config config/binance_spot_example.yaml
```

### 4. Seleccionar Mejor Modelo

```bash
python -m src.cli.main select-model --config config/binance_spot_example.yaml
```

### 5. Ejecutar Backtest

```bash
python -m src.cli.main backtest --config config/binance_spot_example.yaml --walk-forward rolling
```

### 6. Generar Reporte

```bash
python -m src.cli.main report --run-id <run_id>
```

## 📋 Configuración

El sistema usa archivos YAML para configuración. Ejemplo básico:

```yaml
# config/mi_estrategia.yaml
data:
  exchange: binance
  symbol: BTC/USDT
  timeframe: 1h
  start_date: 2023-01-01

models:
  enabled:
    - hmm_gaussian
    - hmm_discrete
    - ms_ar
  
  hmm_gaussian:
    n_components_range: [2, 3, 4]
    covariance_type: full

strategy:
  regime_mapping:
    0: flat     # régimen lateral
    1: long     # régimen alcista  
    2: short    # régimen bajista

risk_management:
  stop_loss:
    method: atr
    atr_multiplier: 1.5
  take_profit:
    method: atr
    atr_multiplier: 3.0
```

## 🏗️ Arquitectura

```
markov-crypto-forecaster/
├── src/
│   ├── data/                 # Descarga y preprocesamiento
│   │   ├── downloader.py     # CCXT/Binance integration
│   │   ├── preprocessor.py   # Limpieza sin look-ahead
│   │   └── feature_engineer.py  # Features técnicas
│   ├── models/               # Modelos de Markov
│   │   ├── hmm_gaussian.py   # HMM Gaussiano
│   │   ├── hmm_discrete.py   # HMM Discreto
│   │   ├── ms_ar.py         # Markov Switching AR
│   │   └── selector.py      # Selección automática
│   ├── forecasting/          # Sistema de pronóstico
│   │   └── forecaster.py    # Predicciones probabilísticas
│   ├── strategy/             # Lógica de trading
│   │   ├── position_logic.py # Señales de trading
│   │   ├── risk.py          # Gestión de riesgo
│   │   ├── backtester.py    # Backtesting engine
│   │   └── metrics.py       # Métricas de performance
│   └── cli/                 # Interfaz de línea de comandos
└── config/                  # Archivos de configuración
```

## 📈 Flujo de Trabajo

### 1. Preparación de Datos
- Descarga de datos OHLCV desde exchanges
- Limpieza y validación temporal
- Splits sin look-ahead bias
- Creación de features técnicas

### 2. Modelado
- Entrenamiento de múltiples modelos de Markov
- Selección automática por criterios de información
- Validación cruzada temporal

### 3. Pronóstico
- Inferencia de régimen actual
- Probabilidades de transición futuras
- Escenarios probabilísticos (P10, P50, P90)

### 4. Trading
- Mapeo de régimen a señales
- Position sizing (Kelly fraction, vol targeting)
- Gestión de riesgo adaptativa

### 5. Evaluación
- Backtesting walk-forward
- Métricas de performance completas
- Reportes visuales automatizados

## 📊 Métricas Incluidas

### Performance
- Total Return, Sharpe Ratio, Sortino Ratio
- Maximum Drawdown, Calmar Ratio
- Profit Factor, Hit Rate
- Average Win/Loss Ratio

### Modelo
- Directional Accuracy
- Brier Score, Log-Loss
- CRPS (Continuous Ranked Probability Score)
- Regime Stability

### Riesgo
- VaR, CVaR
- Exposure Analysis
- Drawdown Distribution

## 🔧 Desarrollo

### Estructura de Tests

```bash
# Ejecutar todos los tests
pytest tests/ -v

# Tests con cobertura
pytest tests/ --cov=src --cov-report=html

# Tests específicos
pytest tests/test_hmm_core.py -v
```

### Linting y Formato

```bash
# Formato de código
black src/ tests/

# Imports
isort src/ tests/

# Linting
flake8 src/ tests/

# Type checking
mypy src/
```

## 🐳 Docker

```bash
# Construir imagen
docker build -t markov-crypto-forecaster .

# Ejecutar contenedor
docker run -v $(pwd)/config:/app/config \
           -v $(pwd)/data:/app/data \
           markov-crypto-forecaster \
           python -m src.cli.main backtest --config config/binance_spot_example.yaml
```

## 📝 Ejemplos de Uso

### Notebook de Inicio Rápido

Ver `notebooks/00_quickstart.ipynb` para un ejemplo completo que:
- Descarga 90 días de datos BTC/USDT 1h
- Entrena HMM con 3 regímenes
- Ejecuta walk-forward backtest
- Genera gráficos de performance

### Uso Programático

```python
from src.data.downloader import ExchangeDataDownloader
from src.models.hmm_gaussian import GaussianHMM
from src.strategy.backtester import WalkForwardBacktester
from src.config_loader import load_config

# Cargar configuración
config = load_config("config/binance_spot_example.yaml")

# Descargar datos
downloader = ExchangeDataDownloader(config)
data = downloader.download_and_save("BTC/USDT", "1h")

# Entrenar modelo
model = GaussianHMM(config, n_components=3)
model.fit(features)

# Backtesting
backtester = WalkForwardBacktester(config)
results = backtester.run_backtest(data, model)
```

## ⚠️ Disclaimers

- **Solo para Fines Educativos**: Este software es para investigación y educación
- **Riesgo Financiero**: El trading implica riesgo de pérdida del capital
- **Sin Garantías**: Los resultados pasados no garantizan performance futura
- **Datos de Mercado**: Usar solo con datos de calidad y exchanges confiables

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork del repositorio
2. Crear branch de feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit de cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push al branch (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

## 📄 Licencia

Este proyecto está bajo licencia MIT. Ver `LICENSE` para más detalles.

## 🙏 Agradecimientos

- **hmmlearn**: Implementación de Hidden Markov Models
- **statsmodels**: Markov Switching Models
- **CCXT**: Unified API para exchanges de criptomonedas
- **TA-Lib**: Indicadores técnicos
- **Typer**: CLI moderna y fácil de usar

---

**Desarrollado con ❤️ por un Senior Quant Engineer**

Para soporte técnico y consultoría en estrategias cuantitativas, contactar al desarrollador.
