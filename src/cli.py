
"""
CLI del proyecto usando Typer.
Comandos principales:
- clean-macosx: limpia artefactos de compresión de macOS.
- backtest-msar: ejecuta un walk-forward con MS-AR sobre un CSV OHLCV.
"""

from __future__ import annotations
import typer
from typing import Optional
from pathlib import Path
import pandas as pd
from rich.console import Console
from rich.table import Table
from datetime import datetime

from .backtest.walkforward import WalkForwardConfig, RiskConfig, walkforward_backtest

app = typer.Typer(add_completion=False)
console = Console()

def _load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # normaliza nombres de columnas
    cols = {c.lower(): c for c in df.columns}
    def pick(name): 
        for k in cols:
            if k == name: 
                return cols[k]
        return None
    required = ["open","high","low","close"]
    for r in required:
        if pick(r) is None:
            raise typer.BadParameter(f"El CSV debe contener columna '{r}'.")
    # timestamp
    ts_col = None
    for cand in ["timestamp","date","datetime","time"]:
        if pick(cand):
            ts_col = pick(cand); break
    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df = df.sort_values(ts_col).set_index(ts_col)
    else:
        df.index = pd.to_datetime(df.index, utc=True)
    # renombra a minúsculas
    df = df.rename(columns={cols.get(k,k):k for k in cols})
    return df

@app.command("clean-macosx")
def clean_macosx(path: Path = typer.Argument(..., help="Ruta raíz del proyecto")):
    "Elimina carpetas __MACOSX y archivos ._*"
    removed = 0
    for p in path.rglob("*"):
        if p.is_dir() and p.name == "__MACOSX":
            import shutil
            shutil.rmtree(p)
            removed += 1
        elif p.is_file() and p.name.startswith("._"):
            p.unlink(missing_ok=True)
            removed += 1
    console.print(f"[green]Listo.[/green] Eliminados {removed} artefactos.")

@app.command("backtest-msar")
def backtest_msar(
    csv: Path = typer.Argument(..., help="CSV con columnas OHLCV y timestamp/date"),
    outdir: Path = typer.Option(Path("outputs"), help="Directorio de salida"),
    regimes: int = typer.Option(2, help="Número de regímenes del MS-AR"),
    prob_threshold: float = typer.Option(0.6, help="Umbral de p_bull para estar long"),
    train_bars: int = typer.Option(2000, help="Barras en ventana de entrenamiento"),
    test_bars: int = typer.Option(250, help="Barras por paso de prueba"),
    atr_window: int = typer.Option(14, help="Ventana ATR para SL/TP"),
    sl_atr: float = typer.Option(1.5, help="SL = ATR * mult"),
    tp_atr: float = typer.Option(3.0, help="TP = ATR * mult"),
    fees_bps: float = typer.Option(2.0, help="Comisión en bps por lado"),
    slippage_bps: float = typer.Option(1.0, help="Slippage bps por lado"),
    allow_short: bool = typer.Option(False, help="Permitir posiciones short"),
):
    """
    Ejecuta backtest walk-forward con MS-AR (statsmodels) y SL/TP por ATR.
    """
    df = _load_csv(csv)
    outdir.mkdir(parents=True, exist_ok=True)

    wf_cfg = WalkForwardConfig(
        regimes=regimes, prob_threshold=prob_threshold, train_bars=train_bars, test_bars=test_bars
    )
    risk = RiskConfig(
        atr_window=atr_window, sl_atr=sl_atr, tp_atr=tp_atr, fees_bps=fees_bps, slippage_bps=slippage_bps,
        allow_short=allow_short
    )

    equity_df, metrics = walkforward_backtest(df, wf_cfg, risk)

    # Guardar outputs
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    csv_out = outdir / f"equity_{stamp}.csv"
    equity_df.to_csv(csv_out, index=True)

    # Render de métricas
    table = Table(title="Resultados Backtest MS-AR")
    for k in ["CAGR","Sharpe","Sortino","MaxDrawdown","FinalEquity","Bars","TradesApprox"]:
        table.add_row(k, f"{metrics[k]:.6f}" if isinstance(metrics[k], float) else str(metrics[k]))
    console.print(table)
    console.print(f"[blue]Equity guardada en:[/blue] {csv_out}")

if __name__ == "__main__":
    app()
