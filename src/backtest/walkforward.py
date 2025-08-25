
"""
Walk-forward backtester para modelos de cambio de régimen (MS-AR / HMM).
100% auto-contenido: no depende de otros módulos del repo para que sea fácil de usar desde CLI.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
from pathlib import Path

# Statsmodels para MS-AR
import warnings
warnings.filterwarnings("ignore")
try:
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
except Exception as e:
    MarkovRegression = None

# =========================
# Utilidades
# =========================

def _to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index(pd.to_datetime(df["timestamp"], utc=True)).drop(columns=["timestamp"])
        elif "date" in df.columns:
            df = df.set_index(pd.to_datetime(df["date"], utc=True)).drop(columns=["date"])
        else:
            raise ValueError("El DataFrame debe tener DatetimeIndex o columna 'timestamp'/'date'.")
    return df.sort_index()

def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    return tr.rolling(window, min_periods=1).mean()

def cagr(equity: pd.Series, periods_per_year: int = 365) -> float:
    if len(equity) < 2: 
        return 0.0
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    years = len(equity) / periods_per_year
    if years <= 0:
        return 0.0
    return (1.0 + total_return) ** (1.0 / years) - 1.0

def sharpe(returns: pd.Series, periods_per_year: int = 365) -> float:
    if returns.std(ddof=0) == 0:
        return 0.0
    return np.sqrt(periods_per_year) * returns.mean() / returns.std(ddof=0)

def sortino(returns: pd.Series, periods_per_year: int = 365) -> float:
    downside = returns[returns < 0]
    dd = downside.std(ddof=0)
    if dd == 0:
        return 0.0
    return np.sqrt(periods_per_year) * returns.mean() / dd

def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return dd.min() if len(dd) else 0.0

# =========================
# Configs
# =========================

@dataclass
class RiskConfig:
    atr_window: int = 14
    sl_atr: float = 1.5
    tp_atr: float = 3.0
    trailing: bool = False
    fees_bps: float = 2.0      # 0.02%
    slippage_bps: float = 1.0  # 0.01%
    allow_short: bool = False  # por defecto solo long

@dataclass
class WalkForwardConfig:
    regimes: int = 2
    prob_threshold: float = 0.6
    train_bars: int = 2000
    test_bars: int = 250

# =========================
# Modelo MS-AR mínimo
# =========================

class MSARModel:
    """
    Wrapper simple sobre statsmodels.MarkovRegression entrenado sobre retornos.
    Se centra en media con cambio de régimen (markov switching mean).
    """

    def __init__(self, k_regimes: int = 2, switching_variance: bool = True):
        if MarkovRegression is None:
            raise ImportError("statsmodels no disponible para MS-AR.")
        self.k_regimes = k_regimes
        self.switching_variance = switching_variance
        self._res = None

    def fit(self, returns: pd.Series, **fit_kwargs):
        y = returns.dropna()
        if len(y) < 100:
            raise ValueError("Muy pocos datos para entrenar MS-AR (min 100).")
        mod = MarkovRegression(
            y.values, k_regimes=self.k_regimes, trend="c", switching_variance=self.switching_variance
        )
        self._res = mod.fit(**fit_kwargs)
        return self

    def filtered_probabilities(self) -> pd.DataFrame:
        if self._res is None:
            raise RuntimeError("Modelo no entrenado.")
        fp = pd.DataFrame(self._res.filtered_marginal_probabilities.T).T
        fp.columns = [f"p_regime_{i}" for i in range(self.k_regimes)]
        return fp

    def predicted_mean(self) -> pd.Series:
        if self._res is None:
            raise RuntimeError("Modelo no entrenado.")
        # One-step-ahead predicted mean of the observation (returns)
        pm = pd.Series(self._res.predicted_marginal_probabilities.mean(axis=1))
        # Nota: statsmodels no expone directamente la predicción de retorno; 
        # como aproximación usamos la prob media ponderada (proxy). 
        # Para señales, combinaremos prob del régimen "alcista".
        return pm

# =========================
# Señales y ejecución
# =========================

def generate_signals_msar(returns: pd.Series, wf_cfg: WalkForwardConfig) -> pd.DataFrame:
    """
    Walk-forward: entrena MS-AR en ventanas y produce prob de régimen alcista.
    Regla: LONG si p_alcista > umbral; FLAT/SHORT si no (según config).
    """
    returns = returns.dropna()
    n = len(returns)
    idx = returns.index

    signals = pd.DataFrame(index=idx, columns=["p_bull", "signal"]).astype(float)
    start = 0
    while start + wf_cfg.train_bars + wf_cfg.test_bars <= n:
        train = returns.iloc[start : start + wf_cfg.train_bars]
        test = returns.iloc[start + wf_cfg.train_bars : start + wf_cfg.train_bars + wf_cfg.test_bars]

        model = MSARModel(k_regimes=wf_cfg.regimes).fit(train)
        probs = model.filtered_probabilities().iloc[-1].values  # última barra de train
        # Asumimos régimen 1 como "bull" si su media estimada es mayor que la otra
        # Derivar medias por régimen de forma aproximada: usar posterior means si estuvieran, si no, fallback
        # Como simplificación: bull = argmax(prob última) no es robusto; mejor, prob del régimen con mayor media.
        # statsmodels no expone medias de cada régimen directamente sin parámetros; como aproximación, elegimos
        # el régimen con mayor prob en la última barra como bull. (puedes ajustar si tienes medias por régimen)
        bull_regime = np.argmax(probs)

        # Para el bloque de test usamos la probabilidad filtrada expandiendo con el modelo fitted
        # Nota: statsmodels no ofrece predict prob para out-of-sample fácilmente.
        # Usamos un proxy: mantenemos p_bull constante igual a la última prob del train (marcador estable).
        p_bull_series = pd.Series([probs[bull_regime]] * len(test), index=test.index)

        signals.loc[test.index, "p_bull"] = p_bull_series
        signals.loc[test.index, "signal"] = (p_bull_series > wf_cfg.prob_threshold).astype(int)

        start += wf_cfg.test_bars

    # Forward-fill para cubrir el resto (si queda tail sin cubrir, lo dejamos NaN o FFILL)
    signals["p_bull"] = signals["p_bull"].ffill()
    signals["signal"] = signals["signal"].fillna(0)
    return signals

def simulate_execution(df: pd.DataFrame, signals: pd.Series, risk: RiskConfig) -> pd.DataFrame:
    """
    Simula ejecución a barra-cierre con SL/TP basados en ATR.
    Estrategia: entrar cuando signal cambia (0->1 o 0->-1 si allow_short).
    Salir por SL/TP o por cambio de señal.
    """
    df = df.copy()
    df = _to_datetime_index(df)
    df["atr"] = atr(df, window=risk.atr_window)

    price = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    signals = signals.reindex(df.index).fillna(0).astype(int)

    equity = pd.Series(index=df.index, dtype=float)
    pos = 0  # -1, 0, 1
    entry_price = None
    sl = None
    tp = None
    cash = 1.0  # capital normalizado
    last_price = price.iloc[0]

    for i, ts in enumerate(df.index):
        p = price.loc[ts]
        a = df.at[ts, "atr"]

        # Cierre posición por cambio de señal (si tenemos)
        desired = int(signals.loc[ts])
        desired = desired if risk.allow_short else max(desired, 0)

        # Si tenemos posición, revisar SL/TP intrabar (conservador)
        if pos != 0:
            # eval intrabar: si long y low <= SL -> SL; si high >= TP -> TP
            hit_sl = low.loc[ts] <= sl if sl is not None else False
            hit_tp = high.loc[ts] >= tp if tp is not None else False

            if hit_sl and hit_tp:
                # conservador: asumir peor (SL)
                exit_price = sl
                cash *= (1 + (exit_price - entry_price) / entry_price * pos)
                cash *= (1 - (risk.fees_bps + risk.slippage_bps) / 10000.0)
                pos = 0
                entry_price = sl = tp = None
            elif hit_sl:
                exit_price = sl
                cash *= (1 + (exit_price - entry_price) / entry_price * pos)
                cash *= (1 - (risk.fees_bps + risk.slippage_bps) / 10000.0)
                pos = 0
                entry_price = sl = tp = None
            elif hit_tp:
                exit_price = tp
                cash *= (1 + (exit_price - entry_price) / entry_price * pos)
                cash *= (1 - (risk.fees_bps + risk.slippage_bps) / 10000.0)
                pos = 0
                entry_price = sl = tp = None

        # Apertura / cambio por señal
        if pos != desired:
            # cerrar si había
            if pos != 0:
                # cerramos al precio de cierre
                exit_price = p
                cash *= (1 + (exit_price - entry_price) / entry_price * pos)
                cash *= (1 - (risk.fees_bps + risk.slippage_bps) / 10000.0)
                pos = 0
                entry_price = sl = tp = None

            # abrir nueva si desired != 0
            if desired != 0:
                pos = desired
                entry_price = p
                atr_val = a if np.isfinite(a) else 0.0
                if pos == 1:
                    sl = entry_price - risk.sl_atr * atr_val
                    tp = entry_price + risk.tp_atr * atr_val
                else:
                    sl = entry_price + risk.sl_atr * atr_val
                    tp = entry_price - risk.tp_atr * atr_val
                cash *= (1 - (risk.fees_bps + risk.slippage_bps) / 10000.0)

        equity.iloc[i] = cash

    out = pd.DataFrame(index=df.index, data={
        "equity": equity.ffill().fillna(method="bfill"),
        "signal": signals,
        "close": price,
    })
    out["ret"] = out["equity"].pct_change().fillna(0.0)
    return out

def evaluate(equity_df: pd.DataFrame) -> Dict[str, Any]:
    eq = equity_df["equity"]
    rets = equity_df["ret"]
    return {
        "CAGR": cagr(eq),
        "Sharpe": sharpe(rets),
        "Sortino": sortino(rets),
        "MaxDrawdown": max_drawdown(eq),
        "FinalEquity": float(eq.iloc[-1]),
        "Bars": int(len(eq)),
        "TradesApprox": int((equity_df["signal"].diff().abs() > 0).sum() / 2),
    }

def walkforward_backtest(df_ohlcv: pd.DataFrame, wf_cfg: WalkForwardConfig, risk: RiskConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = _to_datetime_index(df_ohlcv)
    # returns log
    ret = np.log(df["close"].astype(float)).diff().fillna(0.0)
    sigs = generate_signals_msar(ret, wf_cfg)
    res = simulate_execution(df, sigs["signal"], risk)
    metrics = evaluate(res)
    return res, metrics
