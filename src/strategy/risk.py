"""
Gestión de riesgo para markov-crypto-forecaster.

Implementa stop loss, take profit, trailing stops, gestión de costos
y control de exposición para el sistema de trading.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

from ..config_loader import Config
from ..logging_utils import get_logger, log_execution_time

logger = get_logger(__name__)


class OrderType(Enum):
    """Tipos de órdenes."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Lado de la orden."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Position:
    """Representa una posición abierta."""
    symbol: str
    side: str  # 'long' or 'short'
    size: float  # Tamaño de la posición
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    max_favorable_price: Optional[float] = None  # Para trailing stop
    unrealized_pnl: float = 0.0
    
    def update_unrealized_pnl(self, current_price: float):
        """Actualiza PnL no realizado."""
        if self.side == 'long':
            self.unrealized_pnl = (current_price - self.entry_price) * self.size
        else:  # short
            self.unrealized_pnl = (self.entry_price - current_price) * self.size
        
        # Actualizar precio más favorable para trailing stop
        if self.max_favorable_price is None:
            self.max_favorable_price = current_price
        elif self.side == 'long' and current_price > self.max_favorable_price:
            self.max_favorable_price = current_price
        elif self.side == 'short' and current_price < self.max_favorable_price:
            self.max_favorable_price = current_price


@dataclass
class Trade:
    """Representa un trade completado."""
    symbol: str
    side: str
    size: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    commission: float
    slippage: float
    exit_reason: str  # 'stop_loss', 'take_profit', 'signal', 'trailing_stop'


class RiskManager:
    """
    Gestor de riesgo para el sistema de trading.
    
    Funcionalidades:
    - Stop loss dinámico basado en ATR
    - Take profit con ratios riesgo/recompensa
    - Trailing stops adaptativos
    - Control de exposición máxima
    - Cálculo de costos de transacción
    - Gestión de posiciones múltiples
    """
    
    def __init__(self, config: Union[Config, dict]):
        """
        Inicializa el gestor de riesgo.
        
        Args:
            config: Configuración del sistema
        """
        if isinstance(config, dict):
            self.config = config
        else:
            self.config = config.dict()
            
        self.risk_config = self.config.get('risk_management', {})
        
        # Configuración de stop loss
        self.stop_loss_config = self.risk_config.get('stop_loss', {})
        self.sl_method = self.stop_loss_config.get('method', 'atr')
        self.sl_atr_multiplier = self.stop_loss_config.get('atr_multiplier', 1.5)
        self.sl_percent = self.stop_loss_config.get('percent', 0.02)
        
        # Configuración de take profit
        self.take_profit_config = self.risk_config.get('take_profit', {})
        self.tp_method = self.take_profit_config.get('method', 'atr')
        self.tp_atr_multiplier = self.take_profit_config.get('atr_multiplier', 3.0)
        self.tp_percent = self.take_profit_config.get('percent', 0.04)
        
        # Configuración de trailing stop
        self.trailing_config = self.risk_config.get('trailing_stop', {})
        self.trailing_enabled = self.trailing_config.get('enabled', False)
        self.trailing_atr_multiplier = self.trailing_config.get('atr_multiplier', 2.0)
        
        # Configuración de costos
        self.costs_config = self.risk_config.get('costs', {})
        self.commission_rate = self.costs_config.get('commission_rate', 0.0004)
        self.slippage_rate = self.costs_config.get('slippage_rate', 0.0001)
        self.spread_cost = self.costs_config.get('spread_cost', 0.0001)
        
        # Control de exposición
        self.max_positions = 5  # Máximo número de posiciones simultáneas
        self.max_total_exposure = 1.0  # Máxima exposición total
        
        # Estado interno
        self.positions = {}  # {symbol: Position}
        self.closed_trades = []  # Lista de trades cerrados
        self.capital = 10000.0  # Capital inicial
        self.available_capital = self.capital
        
        logger.info(f"RiskManager inicializado: SL={self.sl_method}, TP={self.tp_method}, "
                   f"trailing={self.trailing_enabled}")
    
    def calculate_stop_loss(self, entry_price: float, 
                           side: str,
                           market_data: pd.DataFrame,
                           atr_series: Optional[pd.Series] = None) -> float:
        """
        Calcula nivel de stop loss.
        
        Args:
            entry_price: Precio de entrada
            side: 'long' o 'short'
            market_data: Datos de mercado recientes
            atr_series: Serie de ATR (opcional)
            
        Returns:
            Precio de stop loss
        """
        if self.sl_method == 'atr':
            if atr_series is not None and len(atr_series) > 0:
                current_atr = atr_series.iloc[-1]
            else:
                # Calcular ATR simple
                current_atr = self._calculate_simple_atr(market_data)
            
            if side == 'long':
                stop_loss = entry_price - (current_atr * self.sl_atr_multiplier)
            else:  # short
                stop_loss = entry_price + (current_atr * self.sl_atr_multiplier)
                
        elif self.sl_method == 'percent':
            if side == 'long':
                stop_loss = entry_price * (1 - self.sl_percent)
            else:  # short
                stop_loss = entry_price * (1 + self.sl_percent)
                
        elif self.sl_method == 'fixed':
            # Valor fijo desde configuración
            fixed_amount = self.stop_loss_config.get('fixed_amount', 10.0)
            if side == 'long':
                stop_loss = entry_price - fixed_amount
            else:  # short
                stop_loss = entry_price + fixed_amount
        else:
            # Sin stop loss
            stop_loss = None
        
        return stop_loss
    
    def calculate_take_profit(self, entry_price: float,
                            side: str, 
                            market_data: pd.DataFrame,
                            atr_series: Optional[pd.Series] = None) -> float:
        """
        Calcula nivel de take profit.
        
        Args:
            entry_price: Precio de entrada
            side: 'long' o 'short'
            market_data: Datos de mercado recientes
            atr_series: Serie de ATR (opcional)
            
        Returns:
            Precio de take profit
        """
        if self.tp_method == 'atr':
            if atr_series is not None and len(atr_series) > 0:
                current_atr = atr_series.iloc[-1]
            else:
                current_atr = self._calculate_simple_atr(market_data)
            
            if side == 'long':
                take_profit = entry_price + (current_atr * self.tp_atr_multiplier)
            else:  # short
                take_profit = entry_price - (current_atr * self.tp_atr_multiplier)
                
        elif self.tp_method == 'percent':
            if side == 'long':
                take_profit = entry_price * (1 + self.tp_percent)
            else:  # short
                take_profit = entry_price * (1 - self.tp_percent)
                
        elif self.tp_method == 'risk_reward':
            # Basado en ratio riesgo/recompensa
            rr_ratio = self.take_profit_config.get('risk_reward_ratio', 2.0)
            
            # Necesitamos el stop loss para calcular el riesgo
            stop_loss = self.calculate_stop_loss(entry_price, side, market_data, atr_series)
            if stop_loss is not None:
                risk = abs(entry_price - stop_loss)
                reward = risk * rr_ratio
                
                if side == 'long':
                    take_profit = entry_price + reward
                else:  # short
                    take_profit = entry_price - reward
            else:
                # Fallback a método percent
                if side == 'long':
                    take_profit = entry_price * (1 + self.tp_percent)
                else:
                    take_profit = entry_price * (1 - self.tp_percent)
        else:
            # Sin take profit
            take_profit = None
        
        return take_profit
    
    def _calculate_simple_atr(self, market_data: pd.DataFrame, period: int = 14) -> float:
        """
        Calcula ATR simple.
        
        Args:
            market_data: Datos OHLC
            period: Período para ATR
            
        Returns:
            Valor ATR actual
        """
        if len(market_data) < period:
            period = len(market_data)
        
        if period < 2:
            return market_data['close'].iloc[-1] * 0.02  # 2% fallback
        
        # True Range
        high_low = market_data['high'] - market_data['low']
        high_prev_close = abs(market_data['high'] - market_data['close'].shift(1))
        low_prev_close = abs(market_data['low'] - market_data['close'].shift(1))
        
        true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
        
        # ATR
        atr = true_range.rolling(window=period).mean().iloc[-1]
        
        return atr if not pd.isna(atr) else market_data['close'].iloc[-1] * 0.02
    
    def calculate_position_size(self, signal_size: float, 
                              entry_price: float,
                              symbol: str = "BTC/USDT") -> float:
        """
        Calcula tamaño de posición considerando capital disponible y riesgo.
        
        Args:
            signal_size: Tamaño sugerido por la estrategia
            entry_price: Precio de entrada
            symbol: Símbolo del activo
            
        Returns:
            Tamaño final de posición
        """
        # Tamaño base sugerido por la estrategia
        base_size = abs(signal_size)
        
        # Verificar capital disponible
        required_capital = base_size * entry_price
        
        if required_capital > self.available_capital:
            # Ajustar tamaño según capital disponible
            adjusted_size = self.available_capital / entry_price
            logger.warning(f"Tamaño ajustado por capital: {base_size:.4f} -> {adjusted_size:.4f}")
            base_size = adjusted_size
        
        # Verificar exposición total
        current_exposure = sum(abs(pos.size * pos.entry_price) for pos in self.positions.values())
        total_exposure = current_exposure + (base_size * entry_price)
        max_allowed_exposure = self.capital * self.max_total_exposure
        
        if total_exposure > max_allowed_exposure:
            available_exposure = max_allowed_exposure - current_exposure
            if available_exposure > 0:
                adjusted_size = available_exposure / entry_price
                logger.warning(f"Tamaño ajustado por exposición: {base_size:.4f} -> {adjusted_size:.4f}")
                base_size = adjusted_size
            else:
                logger.warning("Sin exposición disponible, posición rechazada")
                return 0.0
        
        # Verificar número máximo de posiciones
        if len(self.positions) >= self.max_positions and symbol not in self.positions:
            logger.warning(f"Máximo número de posiciones alcanzado: {self.max_positions}")
            return 0.0
        
        # Aplicar dirección original
        final_size = base_size if signal_size >= 0 else -base_size
        
        return final_size
    
    def calculate_transaction_costs(self, size: float, price: float) -> Dict[str, float]:
        """
        Calcula costos de transacción.
        
        Args:
            size: Tamaño de la transacción
            price: Precio de la transacción
            
        Returns:
            Dict con costos detallados
        """
        notional = abs(size) * price
        
        # Comisión
        commission = notional * self.commission_rate
        
        # Slippage (asumiendo ejecución a mercado)
        slippage = notional * self.slippage_rate
        
        # Spread cost
        spread = notional * self.spread_cost
        
        total_cost = commission + slippage + spread
        
        return {
            'commission': commission,
            'slippage': slippage,
            'spread': spread,
            'total_cost': total_cost,
            'cost_percentage': total_cost / notional if notional > 0 else 0
        }
    
    @log_execution_time("open_position")
    def open_position(self, symbol: str,
                     signal_size: float,
                     entry_price: float,
                     market_data: pd.DataFrame,
                     atr_series: Optional[pd.Series] = None,
                     timestamp: Optional[datetime] = None) -> Optional[Position]:
        """
        Abre una nueva posición.
        
        Args:
            symbol: Símbolo del activo
            signal_size: Tamaño de la señal
            entry_price: Precio de entrada
            market_data: Datos de mercado
            atr_series: Serie de ATR
            timestamp: Timestamp de la operación
            
        Returns:
            Posición creada o None si se rechaza
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Calcular tamaño final
        position_size = self.calculate_position_size(signal_size, entry_price, symbol)
        
        if abs(position_size) < 1e-6:  # Posición muy pequeña o rechazada
            logger.info(f"Posición rechazada para {symbol}: tamaño {position_size}")
            return None
        
        # Determinar lado
        side = 'long' if position_size > 0 else 'short'
        
        # Calcular stop loss y take profit
        stop_loss = self.calculate_stop_loss(entry_price, side, market_data, atr_series)
        take_profit = self.calculate_take_profit(entry_price, side, market_data, atr_series)
        
        # Calcular trailing stop inicial
        trailing_stop = None
        if self.trailing_enabled:
            trailing_stop = self._calculate_trailing_stop(entry_price, side, market_data, atr_series)
        
        # Crear posición
        position = Position(
            symbol=symbol,
            side=side,
            size=abs(position_size),
            entry_price=entry_price,
            entry_time=timestamp,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop=trailing_stop,
            max_favorable_price=entry_price
        )
        
        # Calcular costos
        costs = self.calculate_transaction_costs(position.size, entry_price)
        
        # Actualizar capital disponible
        used_capital = position.size * entry_price
        self.available_capital -= used_capital
        self.available_capital -= costs['total_cost']  # Descontar costos
        
        # Cerrar posición existente si la hay
        if symbol in self.positions:
            logger.info(f"Cerrando posición existente en {symbol}")
            self.close_position(symbol, entry_price, 'signal', timestamp, market_data)
        
        # Guardar nueva posición
        self.positions[symbol] = position
        
        logger.info(f"Posición abierta: {symbol} {side} {position.size:.4f} @ {entry_price:.4f}")
        logger.info(f"SL: {stop_loss:.4f if stop_loss else None}, "
                   f"TP: {take_profit:.4f if take_profit else None}")
        
        return position
    
    def close_position(self, symbol: str,
                      exit_price: float,
                      exit_reason: str,
                      timestamp: Optional[datetime] = None,
                      market_data: Optional[pd.DataFrame] = None) -> Optional[Trade]:
        """
        Cierra una posición existente.
        
        Args:
            symbol: Símbolo del activo
            exit_price: Precio de salida
            exit_reason: Razón del cierre
            timestamp: Timestamp de la operación
            market_data: Datos de mercado (para costos)
            
        Returns:
            Trade completado o None si no había posición
        """
        if symbol not in self.positions:
            logger.warning(f"No hay posición para cerrar en {symbol}")
            return None
        
        if timestamp is None:
            timestamp = datetime.now()
        
        position = self.positions[symbol]
        
        # Calcular PnL
        if position.side == 'long':
            pnl = (exit_price - position.entry_price) * position.size
        else:  # short
            pnl = (position.entry_price - exit_price) * position.size
        
        pnl_pct = pnl / (position.entry_price * position.size) if position.entry_price > 0 else 0
        
        # Calcular costos de cierre
        costs = self.calculate_transaction_costs(position.size, exit_price)
        
        # PnL neto después de costos
        net_pnl = pnl - costs['total_cost']
        
        # Crear trade completado
        trade = Trade(
            symbol=symbol,
            side=position.side,
            size=position.size,
            entry_price=position.entry_price,
            exit_price=exit_price,
            entry_time=position.entry_time,
            exit_time=timestamp,
            pnl=net_pnl,
            pnl_pct=net_pnl / (position.entry_price * position.size),
            commission=costs['commission'],
            slippage=costs['slippage'],
            exit_reason=exit_reason
        )
        
        # Actualizar capital
        returned_capital = position.size * position.entry_price
        self.available_capital += returned_capital + net_pnl
        
        # Guardar trade y limpiar posición
        self.closed_trades.append(trade)
        del self.positions[symbol]
        
        logger.info(f"Posición cerrada: {symbol} PnL={net_pnl:.2f} ({pnl_pct:.2%}) "
                   f"Razón: {exit_reason}")
        
        return trade
    
    def update_positions(self, market_data: Dict[str, pd.DataFrame],
                        atr_data: Optional[Dict[str, pd.Series]] = None) -> List[Dict[str, Any]]:
        """
        Actualiza todas las posiciones y verifica stop loss/take profit.
        
        Args:
            market_data: Datos de mercado por símbolo
            atr_data: Datos de ATR por símbolo
            
        Returns:
            Lista de acciones requeridas
        """
        actions = []
        current_time = datetime.now()
        
        for symbol, position in list(self.positions.items()):
            if symbol not in market_data:
                logger.warning(f"No hay datos de mercado para {symbol}")
                continue
            
            current_price = market_data[symbol]['close'].iloc[-1]
            
            # Actualizar PnL no realizado
            position.update_unrealized_pnl(current_price)
            
            # Verificar stop loss
            if position.stop_loss is not None:
                if ((position.side == 'long' and current_price <= position.stop_loss) or
                    (position.side == 'short' and current_price >= position.stop_loss)):
                    
                    actions.append({
                        'action': 'close_position',
                        'symbol': symbol,
                        'exit_price': position.stop_loss,
                        'exit_reason': 'stop_loss',
                        'timestamp': current_time
                    })
                    continue
            
            # Verificar take profit
            if position.take_profit is not None:
                if ((position.side == 'long' and current_price >= position.take_profit) or
                    (position.side == 'short' and current_price <= position.take_profit)):
                    
                    actions.append({
                        'action': 'close_position',
                        'symbol': symbol,
                        'exit_price': position.take_profit,
                        'exit_reason': 'take_profit',
                        'timestamp': current_time
                    })
                    continue
            
            # Verificar trailing stop
            if self.trailing_enabled and position.trailing_stop is not None:
                # Actualizar trailing stop
                new_trailing = self._update_trailing_stop(
                    position, current_price, market_data[symbol], 
                    atr_data.get(symbol) if atr_data else None
                )
                
                if new_trailing is not None:
                    position.trailing_stop = new_trailing
                
                # Verificar si se activa trailing stop
                if ((position.side == 'long' and current_price <= position.trailing_stop) or
                    (position.side == 'short' and current_price >= position.trailing_stop)):
                    
                    actions.append({
                        'action': 'close_position',
                        'symbol': symbol,
                        'exit_price': position.trailing_stop,
                        'exit_reason': 'trailing_stop',
                        'timestamp': current_time
                    })
                    continue
        
        return actions
    
    def _calculate_trailing_stop(self, entry_price: float,
                               side: str,
                               market_data: pd.DataFrame,
                               atr_series: Optional[pd.Series] = None) -> Optional[float]:
        """Calcula trailing stop inicial."""
        if not self.trailing_enabled:
            return None
        
        if atr_series is not None and len(atr_series) > 0:
            current_atr = atr_series.iloc[-1]
        else:
            current_atr = self._calculate_simple_atr(market_data)
        
        trailing_distance = current_atr * self.trailing_atr_multiplier
        
        if side == 'long':
            return entry_price - trailing_distance
        else:  # short
            return entry_price + trailing_distance
    
    def _update_trailing_stop(self, position: Position,
                            current_price: float,
                            market_data: pd.DataFrame,
                            atr_series: Optional[pd.Series] = None) -> Optional[float]:
        """Actualiza trailing stop según movimiento del precio."""
        if not self.trailing_enabled or position.trailing_stop is None:
            return None
        
        if atr_series is not None and len(atr_series) > 0:
            current_atr = atr_series.iloc[-1]
        else:
            current_atr = self._calculate_simple_atr(market_data)
        
        trailing_distance = current_atr * self.trailing_atr_multiplier
        
        if position.side == 'long':
            # Para long, trailing stop sube con el precio
            new_trailing = current_price - trailing_distance
            return max(position.trailing_stop, new_trailing)
        else:  # short
            # Para short, trailing stop baja con el precio
            new_trailing = current_price + trailing_distance
            return min(position.trailing_stop, new_trailing)
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Obtiene resumen del portfolio.
        
        Returns:
            Resumen con métricas de riesgo y performance
        """
        total_pnl = sum(trade.pnl for trade in self.closed_trades)
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        current_capital = self.available_capital + sum(
            pos.size * pos.entry_price + pos.unrealized_pnl 
            for pos in self.positions.values()
        )
        
        total_return = (current_capital - self.capital) / self.capital
        
        # Exposición actual
        current_exposure = sum(abs(pos.size * pos.entry_price) for pos in self.positions.values())
        exposure_pct = current_exposure / self.capital
        
        return {
            'capital_inicial': self.capital,
            'capital_actual': current_capital,
            'capital_disponible': self.available_capital,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'realized_pnl': total_pnl,
            'unrealized_pnl': unrealized_pnl,
            'n_positions': len(self.positions),
            'n_closed_trades': len(self.closed_trades),
            'current_exposure': current_exposure,
            'exposure_percentage': exposure_pct * 100,
            'max_exposure_limit': self.max_total_exposure * 100,
            'positions': {sym: {
                'side': pos.side,
                'size': pos.size,
                'entry_price': pos.entry_price,
                'unrealized_pnl': pos.unrealized_pnl,
                'stop_loss': pos.stop_loss,
                'take_profit': pos.take_profit
            } for sym, pos in self.positions.items()}
        }


if __name__ == "__main__":
    # Ejemplo de uso
    try:
        from ..config_loader import load_config
        
        config = load_config("config/binance_spot_example.yaml")
        
        # Crear risk manager
        risk_manager = RiskManager(config)
        
        # Simular datos de mercado
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='1H')
        
        prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.02, 100)))
        market_data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, 100)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, 100))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, 100))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, 100)
        }, index=dates)
        
        print(f"Risk Manager configurado:")
        print(f"  Stop Loss: {risk_manager.sl_method} ({risk_manager.sl_atr_multiplier}x ATR)")
        print(f"  Take Profit: {risk_manager.tp_method} ({risk_manager.tp_atr_multiplier}x ATR)")
        print(f"  Trailing Stop: {risk_manager.trailing_enabled}")
        print(f"  Comisión: {risk_manager.commission_rate:.2%}")
        
        # Simular apertura de posición
        entry_price = market_data['close'].iloc[-1]
        signal_size = 0.5  # 50% del capital en long
        
        position = risk_manager.open_position(
            symbol="BTC/USDT",
            signal_size=signal_size,
            entry_price=entry_price,
            market_data=market_data
        )
        
        if position:
            print(f"\nPosición abierta:")
            print(f"  Símbolo: {position.symbol}")
            print(f"  Lado: {position.side}")
            print(f"  Tamaño: {position.size:.4f}")
            print(f"  Precio entrada: {position.entry_price:.2f}")
            print(f"  Stop Loss: {position.stop_loss:.2f if position.stop_loss else None}")
            print(f"  Take Profit: {position.take_profit:.2f if position.take_profit else None}")
        
        # Simular actualización con nuevo precio
        new_price = entry_price * 1.02  # 2% ganancia
        mock_market_data = {"BTC/USDT": market_data}
        
        actions = risk_manager.update_positions(mock_market_data)
        
        # Mostrar resumen del portfolio
        portfolio = risk_manager.get_portfolio_summary()
        print(f"\nResumen del Portfolio:")
        print(f"  Capital inicial: ${portfolio['capital_inicial']:.2f}")
        print(f"  Capital actual: ${portfolio['capital_actual']:.2f}")
        print(f"  Return total: {portfolio['total_return_pct']:.2f}%")
        print(f"  Posiciones abiertas: {portfolio['n_positions']}")
        print(f"  Exposición: {portfolio['exposure_percentage']:.1f}%")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
