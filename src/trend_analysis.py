import os
import requests
import numpy as np
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple
from supabase import ClientOptions, create_client, Client
from schemas import Trend, PriceData

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BINANCE_API_URL = os.getenv("BINANCE_API_URL", "https://api.binance.com/api/v3")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY, options=ClientOptions(schema="public"))

class TrendAnalyzer:
    def __init__(self):
        self.binance_api_url = BINANCE_API_URL

    def fetch_historical_prices(self, symbol: str, limit: int = 100, interval: str = "15m") -> Tuple[List[Decimal], List[Decimal], List[Decimal], List[Decimal], List[float]]:
        """
        Fetch historical OHLCV data from Binance API
        Returns: (opens, highs, lows, closes, volumes)
        """
        try:
            url = f"{self.binance_api_url}/klines"
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            # Extract OHLCV data
            opens = [Decimal(str(kline[1])) for kline in data]
            highs = [Decimal(str(kline[2])) for kline in data]
            lows = [Decimal(str(kline[3])) for kline in data]
            closes = [Decimal(str(kline[4])) for kline in data]
            volumes = [float(kline[5]) for kline in data]  # Volume in base asset

            return opens, highs, lows, closes, volumes
        except Exception as e:
            print(f"Error fetching historical prices for {symbol}: {e}")
            return [], [], [], [], []

    def calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return []

        ema = []
        multiplier = 2 / (period + 1)

        # Start with SMA for first value
        sma = sum(prices[:period]) / period
        ema.append(sma)

        # Calculate EMA for remaining values
        for price in prices[period:]:
            ema_value = (price - ema[-1]) * multiplier + ema[-1]
            ema.append(ema_value)

        return ema

    def calculate_atr(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> List[float]:
        """Calculate Average True Range"""
        if len(highs) < period + 1:
            return []

        true_ranges = []
        for i in range(1, len(highs)):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i-1])
            low_close = abs(lows[i] - closes[i-1])
            true_range = max(high_low, high_close, low_close)
            true_ranges.append(true_range)

        # Calculate ATR using EMA smoothing
        atr = []
        atr_value = sum(true_ranges[:period]) / period
        atr.append(atr_value)

        for tr in true_ranges[period:]:
            atr_value = (tr + (period - 1) * atr[-1]) / period
            atr.append(atr_value)

        return atr

    def calculate_adx(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Tuple[float, float, float]:
        """
        Calculate proper ADX (Average Directional Index) with +DI and -DI
        Returns: (ADX value, +DI, -DI)
        ADX Scale:
        0-25: Weak/No trend (sideways)
        25-50: Strong trend
        50-75: Very strong trend
        75-100: Extremely strong trend
        """
        if len(highs) < period + 1:
            return 0.0, 0.0, 0.0

        # Calculate +DM and -DM
        plus_dm = []
        minus_dm = []

        for i in range(1, len(highs)):
            high_diff = highs[i] - highs[i-1]
            low_diff = lows[i-1] - lows[i]

            plus_dm.append(high_diff if high_diff > low_diff and high_diff > 0 else 0)
            minus_dm.append(low_diff if low_diff > high_diff and low_diff > 0 else 0)

        # Calculate True Range
        true_ranges = []
        for i in range(1, len(highs)):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i-1])
            low_close = abs(lows[i] - closes[i-1])
            true_ranges.append(max(high_low, high_close, low_close))

        # Smooth +DM, -DM, and TR using Wilder's smoothing
        smoothed_plus_dm = sum(plus_dm[:period])
        smoothed_minus_dm = sum(minus_dm[:period])
        smoothed_tr = sum(true_ranges[:period])

        for i in range(period, len(plus_dm)):
            smoothed_plus_dm = smoothed_plus_dm - (smoothed_plus_dm / period) + plus_dm[i]
            smoothed_minus_dm = smoothed_minus_dm - (smoothed_minus_dm / period) + minus_dm[i]
            smoothed_tr = smoothed_tr - (smoothed_tr / period) + true_ranges[i]

        # Calculate +DI and -DI
        plus_di = 100 * (smoothed_plus_dm / smoothed_tr) if smoothed_tr != 0 else 0
        minus_di = 100 * (smoothed_minus_dm / smoothed_tr) if smoothed_tr != 0 else 0

        # Calculate DX
        di_sum = plus_di + minus_di
        di_diff = abs(plus_di - minus_di)
        dx = 100 * (di_diff / di_sum) if di_sum != 0 else 0

        # Calculate ADX (smoothed DX)
        # For simplicity, we'll use the current DX as ADX
        # In production, you'd smooth this over 14 periods
        adx = dx

        return adx, plus_di, minus_di

    def calculate_macd(self, prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        Returns: (MACD line, Signal line, Histogram)
        """
        if len(prices) < slow + signal:
            return 0.0, 0.0, 0.0

        # Calculate EMAs
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)

        if not ema_fast or not ema_slow:
            return 0.0, 0.0, 0.0

        # Align the EMAs (slow EMA starts later)
        offset = slow - fast
        ema_fast = ema_fast[offset:]

        # Calculate MACD line
        macd_line = [ema_fast[i] - ema_slow[i] for i in range(len(ema_slow))]

        # Calculate Signal line (EMA of MACD)
        if len(macd_line) < signal:
            return 0.0, 0.0, 0.0

        signal_line = self.calculate_ema(macd_line, signal)

        if not signal_line:
            return 0.0, 0.0, 0.0

        # Calculate Histogram
        macd_current = macd_line[-1]
        signal_current = signal_line[-1]
        histogram = macd_current - signal_current

        return macd_current, signal_current, histogram

    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """
        Calculate RSI (Relative Strength Index)
        Returns: RSI value (0-100)

        Interpretation:
        - Above 70: Overbought (potential reversal down)
        - Below 30: Oversold (potential reversal up)
        - 50: Neutral
        """
        if len(prices) < period + 1:
            return 50.0

        # Calculate price changes
        deltas = np.diff(prices)

        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        # Calculate average gains and losses
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        # Smooth using Wilder's method
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        # Calculate RS and RSI
        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_bollinger_bands(self, closes: List[float], period: int = 20, multiplier: float = 2.0) -> Dict:
        """
        Calculate Bollinger Bands with additional indicators for trading signals

        Parameters:
        - closes: List of closing prices
        - period: Number of periods for SMA calculation (default: 20)
        - multiplier: Standard deviation multiplier for band width (default: 2.0)

        Returns: Dict containing bands, bandwidth, %B, and trading signals
        """
        if len(closes) < period:
            return {}

        arr = np.array(closes, dtype=float)

        # Simple Moving Average (middle band) using valid convolution
        sma_series = np.convolve(arr, np.ones(period) / period, mode="valid")

        # Standard deviation series (population std, ddof=0) for each window
        std_series = np.array([arr[i : i + period].std(ddof=0) for i in range(len(arr) - period + 1)])

        upper_band = sma_series + multiplier * std_series
        lower_band = sma_series - multiplier * std_series

        # Current (most recent) values
        current_sma = float(sma_series[-1])
        current_upper = float(upper_band[-1])
        current_lower = float(lower_band[-1])
        current_price = float(arr[-1])

        # Bandwidth (%)
        bandwidth = ((current_upper - current_lower) / current_sma) * 100 if current_sma != 0 else 0.0

        # %B indicator (clamped 0..1 for within-band, <0 or >1 if outside)
        percent_b = (
            (current_price - current_lower) / (current_upper - current_lower)
            if (current_upper - current_lower) != 0
            else 0.0
        )

        # Improved squeeze detection:
        # Use the last up to 100 bandwidth values and consider a squeeze when the current
        # bandwidth is below the 20th percentile and also significantly below the median.
        squeeze = False
        squeeze_info = {"percentile_20": None, "median": None}
        window = min(100, len(sma_series))
        if window > 1:
            hist_bw = ((upper_band[-window:] - lower_band[-window:]) / sma_series[-window:]) * 100
            percentile_20 = float(np.percentile(hist_bw, 20))
            median_bw = float(np.median(hist_bw))
            squeeze_info["percentile_20"] = percentile_20
            squeeze_info["median"] = median_bw
            # require both very low relative to historical and an absolute small value to avoid false positives
            squeeze = (bandwidth < percentile_20) and (bandwidth < max(median_bw * 0.75, 0.5))
        else:
            squeeze = False

        # Band walk detection (trending if price outside bands)
        band_walk = ""
        if current_price > current_upper:
            band_walk = "upper"
        elif current_price < current_lower:
            band_walk = "lower"

        # Mean reversion signal (price near bands inside range)
        mean_reversion_signal = ""
        # Simple thresholds for mean reversion: near lower band -> buy, near upper band -> sell
        if 0 <= percent_b <= 0.1:
            mean_reversion_signal = "buy"
        elif 0.9 <= percent_b <= 1.0:
            mean_reversion_signal = "sell"

        # Simpler breakout signals:
        # - breakout 'up' when price clears upper band and bandwidth is expanding versus recent median
        # - breakout 'down' when price clears lower band and bandwidth is expanding versus recent median
        breakout_signal = ""
        if window > 1 and squeeze_info["median"] is not None:
            if (current_price > current_upper) and (bandwidth > max(squeeze_info["median"] * 1.2, 0.5)):
                breakout_signal = "up"
            elif (current_price < current_lower) and (bandwidth > max(squeeze_info["median"] * 1.2, 0.5)):
                breakout_signal = "down"
        else:
            # fallback: if price is clearly outside bands, mark a breakout without bandwidth confirmation
            if current_price > current_upper:
                breakout_signal = "up"
            elif current_price < current_lower:
                breakout_signal = "down"

        # Simplified M-top / W-bottom detection (placeholder, kept minimal)
        m_top = False
        w_bottom = False
        if len(closes) >= 5:
            last5 = [float(x) for x in closes[-5:]]
            mid = last5[2]
            # M-top: local peaks around center
            if last5[1] > mid and last5[3] > mid and mid < last5[0] and mid < last5[4]:
                m_top = True
            # W-bottom: local troughs around center
            if last5[1] < mid and last5[3] < mid and mid > last5[0] and mid > last5[4]:
                w_bottom = True

        return {
            "middle_band": current_sma,
            "upper_band": current_upper,
            "lower_band": current_lower,
            "bandwidth": bandwidth,
            "percent_b": percent_b,
            "squeeze": squeeze,
            "band_walk": band_walk,
            "mean_reversion_signal": mean_reversion_signal,
            "breakout_signal": breakout_signal,
            "m_top": m_top,
            "w_bottom": w_bottom,
            "period": period,
            "multiplier": multiplier,
        }

    def calculate_volume_trend(self, volumes: List[float], period: int = 20) -> Tuple[str, float]:
        """
        Analyze volume trend
        Returns: (trend, ratio)

        - "increasing": Recent volume > average (bullish confirmation)
        - "decreasing": Recent volume < average (weak trend)
        - "neutral": Volume around average
        """
        if len(volumes) < period:
            return "neutral", 1.0

        avg_volume = np.mean(volumes[-period:])
        recent_volume = np.mean(volumes[-5:])  # Last 5 periods

        if avg_volume == 0:
            return "neutral", 1.0

        ratio = recent_volume / avg_volume

        if ratio > 1.2:
            return "increasing", ratio
        elif ratio < 0.8:
            return "decreasing", ratio
        else:
            return "neutral", ratio

    def find_support_resistance(self, highs: List[float], lows: List[float], closes: List[float], num_levels: int = 3) -> Dict:
        """
        Find key support and resistance levels using pivot points and price clusters
        Returns: Dict with support and resistance levels
        """
        if len(closes) < 20:
            return {"support": [], "resistance": [], "current_price": closes[-1] if closes else 0}

        current_price = closes[-1]

        # Method 1: Recent swing highs and lows
        swing_highs = []
        swing_lows = []

        for i in range(2, len(highs) - 2):
            # Swing high: higher than 2 bars before and after
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                swing_highs.append(highs[i])

            # Swing low: lower than 2 bars before and after
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                swing_lows.append(lows[i])

        # Method 2: Round numbers (psychological levels)
        price_magnitude = 10 ** (len(str(int(current_price))) - 2)
        round_levels = []
        for mult in range(1, 20):
            level = round(current_price / price_magnitude) * price_magnitude
            round_levels.append(level + (mult * price_magnitude))
            round_levels.append(level - (mult * price_magnitude))

        # Combine and filter
        all_resistance = [h for h in swing_highs if h > current_price] + [r for r in round_levels if r > current_price]
        all_support = [l for l in swing_lows if l < current_price] + [s for s in round_levels if s < current_price]

        # Cluster nearby levels (within 1%)
        def cluster_levels(levels, tolerance=0.01):
            if not levels:
                return []

            levels = sorted(levels)
            clusters = []
            current_cluster = [levels[0]]

            for level in levels[1:]:
                if current_cluster[-1] != 0 and abs(level - current_cluster[-1]) / abs(current_cluster[-1]) < tolerance:
                    current_cluster.append(level)
                else:
                    clusters.append(np.mean(current_cluster))
                    current_cluster = [level]

            clusters.append(np.mean(current_cluster))
            return clusters

        resistance_levels = cluster_levels(all_resistance)[:num_levels]
        support_levels = cluster_levels(all_support)[:num_levels]

        # Sort by proximity to current price
        resistance_levels = sorted(resistance_levels)[:num_levels]
        support_levels = sorted(support_levels, reverse=True)[:num_levels]

        # Filter out non-positive levels to avoid negative or zero price levels
        support_levels = [s for s in support_levels if s is not None and s > 0]
        resistance_levels = [r for r in resistance_levels if r is not None and r > 0]

        # Ensure sorted order and limit to requested number of levels
        resistance_levels = sorted(resistance_levels)[:num_levels]
        support_levels = sorted(support_levels, reverse=True)[:num_levels]

        return {
            "support": support_levels,
            "resistance": resistance_levels,
            "current_price": current_price,
            "nearest_support": support_levels[0] if support_levels else None,
            "nearest_resistance": resistance_levels[0] if resistance_levels else None
        }

    def calculate_risk_reward(self, entry_price: float, stop_loss: float, take_profit: float) -> Dict:
        """
        Calculate risk/reward ratio and position sizing
        Returns: Dict with risk/reward metrics
        """
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        rr_ratio = reward / risk if risk > 0 else float('inf')

        # Default position sizing parameters
        # These could be made configurable elsewhere in the class
        account_risk_pct = 0.01  # risk 1% of account per trade
        account_balance = 10000  # placeholder, should be provided externally

        risk_amount = account_balance * account_risk_pct
        position_size = (risk_amount / risk) if risk > 0 else 0

        return {
            "risk": risk,
            "reward": reward,
            "risk_reward_ratio": rr_ratio,
            "position_size": position_size,
            "risk_amount": risk_amount
        }

    def find_swing_points(self, highs: List[float], lows: List[float], lookback: int = 5) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """
        Find swing highs and lows using pivot point analysis
        A swing high is higher than N bars before and after
        A swing low is lower than N bars before and after

        Returns: (swing_highs, swing_lows) as lists of (index, price) tuples
        """
        swing_highs = []
        swing_lows = []

        # Need at least lookback*2 + 1 bars to identify a swing
        if len(highs) < lookback * 2 + 1:
            return swing_highs, swing_lows

        # Find swing highs
        for i in range(lookback, len(highs) - lookback):
            is_swing_high = True
            for j in range(1, lookback + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_swing_high = False
                    break

            if is_swing_high:
                swing_highs.append((i, highs[i]))

        # Find swing lows
        for i in range(lookback, len(lows) - lookback):
            is_swing_low = True
            for j in range(1, lookback + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_swing_low = False
                    break

            if is_swing_low:
                swing_lows.append((i, lows[i]))

        return swing_highs, swing_lows

    def find_significant_swing(self, highs: List[float], lows: List[float],
                              lookback: int = 100, min_move_pct: float = 5.0) -> Optional[Dict]:
        """
        Find the most recent significant swing (highest high and lowest low)
        Only considers swings with minimum price movement to filter noise

        Args:
            highs: List of high prices
            lows: List of low prices
            lookback: Number of candles to look back (default 100)
            min_move_pct: Minimum % price movement to qualify as significant (default 5%)

        Returns: Dict with swing_high, swing_low, swing_type, and indices
        """
        if len(highs) < lookback or len(lows) < lookback:
            lookback = min(len(highs), len(lows))

        if lookback < 10:
            return None

        # Get recent data
        recent_highs = highs[-lookback:]
        recent_lows = lows[-lookback:]

        # Find highest high and lowest low
        swing_high = max(recent_highs)
        swing_low = min(recent_lows)
        swing_high_idx = len(highs) - lookback + recent_highs.index(swing_high)
        swing_low_idx = len(lows) - lookback + recent_lows.index(swing_low)

        # Calculate price movement
        price_range = swing_high - swing_low
        move_pct = (price_range / swing_low) * 100 if swing_low > 0 else 0

        # Check if movement is significant enough
        if move_pct < min_move_pct:
            return None

        # Determine swing type (uptrend if low came before high)
        swing_type = "bullish" if swing_low_idx < swing_high_idx else "bearish"

        return {
            "swing_high": swing_high,
            "swing_low": swing_low,
            "swing_high_idx": swing_high_idx,
            "swing_low_idx": swing_low_idx,
            "swing_type": swing_type,
            "price_range": price_range,
            "move_pct": move_pct
        }

    def calculate_fibonacci_levels(self, swing_high: float, swing_low: float,
                                   swing_type: str = "bullish") -> Dict:
        """
        Calculate Fibonacci retracement and extension levels

        Retracement levels (0%, 23.6%, 38.2%, 50%, 61.8%, 78.6%, 100%)
        Extension levels (127.2%, 161.8%, 200%, 261.8%)

        Args:
            swing_high: The swing high price
            swing_low: The swing low price
            swing_type: "bullish" (uptrend) or "bearish" (downtrend)

        Returns: Dict with retracement and extension levels
        """
        price_range = swing_high - swing_low

        # Fibonacci ratios
        fib_ratios = {
            "0.0": 0.0,      # 0%
            "23.6": 0.236,   # 23.6%
            "38.2": 0.382,   # 38.2% - Key level
            "50.0": 0.500,   # 50% - Key level
            "61.8": 0.618,   # 61.8% - Golden ratio (most important)
            "78.6": 0.786,   # 78.6% - Last chance entry
            "100.0": 1.000   # 100%
        }

        # Extension ratios for profit targets
        extension_ratios = {
            "127.2": 1.272,  # 127.2%
            "161.8": 1.618,  # 161.8% - Golden ratio extension
            "200.0": 2.000,  # 200%
            "261.8": 2.618   # 261.8%
        }

        retracements = {}
        extensions = {}

        if swing_type == "bullish":
            # For uptrends: retracements go down from swing_high
            for name, ratio in fib_ratios.items():
                retracements[name] = swing_high - (price_range * ratio)

            # Extensions go up from swing_high
            for name, ratio in extension_ratios.items():
                extensions[name] = swing_high + (price_range * (ratio - 1))

        else:  # bearish
            # For downtrends: retracements go up from swing_low
            for name, ratio in fib_ratios.items():
                retracements[name] = swing_low + (price_range * ratio)

            # Extensions go down from swing_low
            for name, ratio in extension_ratios.items():
                extensions[name] = swing_low - (price_range * (ratio - 1))

        return {
            "retracements": retracements,
            "extensions": extensions,
            "swing_high": swing_high,
            "swing_low": swing_low,
            "swing_type": swing_type,
            "price_range": price_range
        }

    def find_nearest_fib_level(self, current_price: float, fib_levels: Dict,
                               tolerance: float = 0.005) -> Optional[Dict]:
        """
        Find which Fibonacci level the current price is near

        Args:
            current_price: Current market price
            fib_levels: Dict from calculate_fibonacci_levels
            tolerance: Price tolerance as decimal (0.005 = 0.5%)

        Returns: Dict with nearest level info or None
        """
        retracements = fib_levels.get("retracements", {})
        extensions = fib_levels.get("extensions", {})

        all_levels = {}
        for name, price in retracements.items():
            all_levels[f"Fib {name}%"] = price
        for name, price in extensions.items():
            all_levels[f"Ext {name}%"] = price

        nearest_level = None
        min_distance = float('inf')

        for level_name, level_price in all_levels.items():
            distance = abs(current_price - level_price)
            distance_pct = distance / current_price if current_price > 0 else float('inf')

            if distance_pct <= tolerance and distance < min_distance:
                min_distance = distance
                nearest_level = {
                    "level_name": level_name,
                    "level_price": level_price,
                    "distance": distance,
                    "distance_pct": distance_pct * 100,
                    "is_key_level": any(key in level_name for key in ["38.2", "50.0", "61.8"])
                }

        return nearest_level

    def detect_fibonacci_confluence(self, current_price: float, fib_levels: Dict,
                                    sr_levels: Dict, ema_levels: Dict = None,
                                    tolerance: float = 0.01) -> List[Dict]:
        """
        Detect confluence zones where Fibonacci levels align with other indicators

        Confluence increases probability of support/resistance holding

        Args:
            current_price: Current market price
            fib_levels: Fibonacci levels dict
            sr_levels: Support/resistance levels dict
            ema_levels: Dict with EMA values (optional)
            tolerance: Confluence tolerance (1% default)

        Returns: List of confluence zones with details
        """
        confluences = []

        # Get all Fibonacci levels
        all_fib_levels = {}
        for name, price in fib_levels.get("retracements", {}).items():
            all_fib_levels[f"Fib {name}%"] = price
        for name, price in fib_levels.get("extensions", {}).items():
            all_fib_levels[f"Ext {name}%"] = price

        # Check each Fibonacci level for confluence
        for fib_name, fib_price in all_fib_levels.items():
            confluence_factors = [fib_name]

            # Check support/resistance alignment
            for support in sr_levels.get("support", []):
                if abs(fib_price - support) / fib_price <= tolerance:
                    confluence_factors.append("Support Level")
                    break

            for resistance in sr_levels.get("resistance", []):
                if abs(fib_price - resistance) / fib_price <= tolerance:
                    confluence_factors.append("Resistance Level")
                    break

            # Check EMA alignment
            if ema_levels:
                for ema_name, ema_value in ema_levels.items():
                    if ema_value and abs(fib_price - ema_value) / fib_price <= tolerance:
                        confluence_factors.append(f"{ema_name}")

            # Check round number (psychological level)
            # Round to nearest 100, 1000, 10000 depending on price magnitude
            magnitude = 10 ** (len(str(int(fib_price))) - 2)
            round_price = round(fib_price / magnitude) * magnitude
            if abs(fib_price - round_price) / fib_price <= tolerance:
                confluence_factors.append("Round Number")

            # If we have confluence (2+ factors), add to list
            if len(confluence_factors) >= 2:
                distance_from_current = abs(current_price - fib_price) / current_price * 100

                confluences.append({
                    "price": fib_price,
                    "factors": confluence_factors,
                    "confluence_count": len(confluence_factors),
                    "distance_pct": distance_from_current,
                    "is_key_level": any(key in fib_name for key in ["38.2", "50.0", "61.8"])
                })

        # Sort by confluence count (strongest first)
        confluences.sort(key=lambda x: (-x["confluence_count"], x["distance_pct"]))

        return confluences

    def analyze_fibonacci(self, symbol: str, lookback: int = 100,
                         min_move_pct: float = 5.0, interval: str = "15m") -> Optional[Dict]:
        """
        Comprehensive Fibonacci analysis for a trading pair

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            lookback: Candles to look back for swing detection
            min_move_pct: Minimum % move to qualify as significant swing
            interval: Timeframe (15m, 1h, 4h, etc.)

        Returns: Complete Fibonacci analysis dict
        """
        # Fetch historical data
        opens, highs, lows, closes, volumes = self.fetch_historical_prices(symbol, lookback + 50, interval)

        if not closes or len(closes) < 50:
            return None

        # Convert to float
        highs_float = [float(h) for h in highs]
        lows_float = [float(l) for l in lows]
        closes_float = [float(c) for c in closes]
        current_price = closes_float[-1]

        # Find significant swing
        swing = self.find_significant_swing(highs_float, lows_float, lookback, min_move_pct)

        if not swing:
            return {
                "symbol": symbol,
                "interval": interval,
                "error": "No significant swing found",
                "current_price": current_price
            }

        # Calculate Fibonacci levels
        fib_levels = self.calculate_fibonacci_levels(
            swing["swing_high"],
            swing["swing_low"],
            swing["swing_type"]
        )

        # Find nearest Fibonacci level
        nearest_fib = self.find_nearest_fib_level(current_price, fib_levels)

        # Get support/resistance for confluence
        sr_levels = self.find_support_resistance(highs_float, lows_float, closes_float)

        # Get EMA levels for confluence
        ema_9 = self.calculate_ema(closes_float, 9)
        ema_21 = self.calculate_ema(closes_float, 21)
        ema_50 = self.calculate_ema(closes_float, 50)

        ema_levels = {
            "EMA 9": ema_9[-1] if ema_9 else None,
            "EMA 21": ema_21[-1] if ema_21 else None,
            "EMA 50": ema_50[-1] if ema_50 else None
        }

        # Detect confluence zones
        confluences = self.detect_fibonacci_confluence(
            current_price, fib_levels, sr_levels, ema_levels
        )

        # Determine trading zones
        retracements = fib_levels["retracements"]
        extensions = fib_levels["extensions"]

        # Optimal entry zones (38.2% - 61.8%)
        entry_zone = {
            "optimal_entry_low": retracements.get("38.2", 0),
            "optimal_entry_high": retracements.get("61.8", 0),
            "last_chance_entry": retracements.get("78.6", 0),
            "invalidation": retracements.get("100.0", 0)
        }

        # Profit targets (extensions)
        profit_targets = {
            "target_1": extensions.get("127.2", 0),  # Take 50%
            "target_2": extensions.get("161.8", 0),  # Take 25%
            "target_3": extensions.get("200.0", 0),  # Let 25% run
            "target_4": extensions.get("261.8", 0)   # Moon target
        }

        # Analyze current position relative to Fibonacci
        position_analysis = self._analyze_fib_position(
            current_price, retracements, extensions, swing["swing_type"]
        )

        return {
            "symbol": symbol,
            "interval": interval,
            "current_price": current_price,
            "swing": swing,
            "fibonacci_levels": fib_levels,
            "nearest_fib_level": nearest_fib,
            "entry_zone": entry_zone,
            "profit_targets": profit_targets,
            "confluences": confluences,
            "position_analysis": position_analysis,
            "ema_levels": ema_levels,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def _analyze_fib_position(self, current_price: float, retracements: Dict,
                              extensions: Dict, swing_type: str) -> Dict:
        """
        Analyze where current price is relative to Fibonacci levels
        """
        analysis = {
            "in_entry_zone": False,
            "at_key_level": False,
            "near_invalidation": False,
            "at_profit_target": False,
            "recommendation": ""
        }

        # Check if in optimal entry zone (38.2% - 61.8%)
        fib_382 = retracements.get("38.2", 0)
        fib_618 = retracements.get("61.8", 0)
        fib_786 = retracements.get("78.6", 0)
        fib_100 = retracements.get("100.0", 0)

        if swing_type == "bullish":
            # In bullish retracement, lower prices are deeper retracements
            if fib_618 <= current_price <= fib_382:
                analysis["in_entry_zone"] = True
                analysis["recommendation"] = "🎯 OPTIMAL ENTRY ZONE - Consider entering LONG"
            elif fib_786 <= current_price < fib_618:
                analysis["in_entry_zone"] = True
                analysis["recommendation"] = "⚠️ DEEP RETRACEMENT - Last chance entry with tight stop"
            elif current_price < fib_786:
                analysis["near_invalidation"] = True
                analysis["recommendation"] = "🚫 NEAR INVALIDATION - Wait for reversal confirmation"
        else:  # bearish
            # In bearish retracement, higher prices are deeper retracements
            if fib_382 <= current_price <= fib_618:
                analysis["in_entry_zone"] = True
                analysis["recommendation"] = "🎯 OPTIMAL ENTRY ZONE - Consider entering SHORT"
            elif fib_618 < current_price <= fib_786:
                analysis["in_entry_zone"] = True
                analysis["recommendation"] = "⚠️ DEEP RETRACEMENT - Last chance entry with tight stop"
            elif current_price > fib_786:
                analysis["near_invalidation"] = True
                analysis["recommendation"] = "🚫 NEAR INVALIDATION - Wait for reversal confirmation"

        # Check if at key level (within 0.5%)
        key_levels = [fib_382, fib_618, retracements.get("50.0", 0)]
        for level in key_levels:
            if level and abs(current_price - level) / current_price <= 0.005:
                analysis["at_key_level"] = True
                break

        # Check if at profit target
        for ext_name, ext_price in extensions.items():
            if ext_price and abs(current_price - ext_price) / current_price <= 0.005:
                analysis["at_profit_target"] = True
                analysis["recommendation"] = f"💰 AT PROFIT TARGET ({ext_name}%) - Consider taking profits"
                break

        return analysis

    def multi_timeframe_fibonacci(self, symbol: str) -> Dict:
        """
        Analyze Fibonacci levels across multiple timeframes
        Higher timeframe Fibonacci levels are more significant

        Returns: Dict with Fibonacci analysis for each timeframe
        """
        timeframes = {
            "15m": "15m",   # Entry timeframe
            "1h": "1h",     # Confirmation timeframe
            "4h": "4h"      # Major levels timeframe
        }

        results = {}

        for name, interval in timeframes.items():
            fib_analysis = self.analyze_fibonacci(symbol, lookback=100, interval=interval)

            if fib_analysis and "error" not in fib_analysis:
                results[name] = {
                    "swing_type": fib_analysis["swing"]["swing_type"],
                    "move_pct": fib_analysis["swing"]["move_pct"],
                    "nearest_fib": fib_analysis.get("nearest_fib_level"),
                    "in_entry_zone": fib_analysis["position_analysis"]["in_entry_zone"],
                    "confluences": len(fib_analysis.get("confluences", [])),
                    "key_levels": {
                        "fib_382": fib_analysis["fibonacci_levels"]["retracements"].get("38.2"),
                        "fib_50": fib_analysis["fibonacci_levels"]["retracements"].get("50.0"),
                        "fib_618": fib_analysis["fibonacci_levels"]["retracements"].get("61.8")
                    }
                }
            else:
                results[name] = {"error": "Insufficient data or no significant swing"}

        # Check for multi-timeframe alignment
        swing_types = [r.get("swing_type") for r in results.values() if "swing_type" in r]
        aligned = len(set(swing_types)) == 1 if swing_types else False

        results["alignment"] = {
            "is_aligned": aligned,
            "consensus_swing": swing_types[0] if aligned and swing_types else "mixed"
        }

        return results

    def get_fibonacci_trade_setup(self, symbol: str, interval: str = "15m", 
                                 account_balance: float = 10000.0, 
                                 risk_percent: float = 1.0) -> Optional[Dict]:
        """
        Generate complete Fibonacci trade setup with entry, stop loss, and take profits
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Timeframe (15m, 1h, 4h, etc.)
            account_balance: Account balance for position sizing
            risk_percent: Risk percentage per trade (default 1%)
            
        Returns: Complete trade setup dict
        """
        # Get Fibonacci analysis
        # Use lower min_move_pct for shorter timeframes
        min_move_pct = 2.0 if interval in ["15m", "5m", "1m"] else 5.0
        fib_analysis = self.analyze_fibonacci(symbol, interval=interval, min_move_pct=min_move_pct)
        
        if not fib_analysis or "error" in fib_analysis:
            return None
            
        current_price = fib_analysis["current_price"]
        swing = fib_analysis["swing"]
        position_analysis = fib_analysis["position_analysis"]
        retracements = fib_analysis["fibonacci_levels"]["retracements"]
        extensions = fib_analysis["fibonacci_levels"]["extensions"]
        
        # Determine trade direction
        if swing["swing_type"] == "bullish":
            direction = "LONG"
            # For bullish, we buy retracements and target extensions
            entry_price = current_price
            stop_loss = retracements.get("100.0", swing["swing_low"])  # Below swing low
            
            # Take profit levels
            tp1_price = extensions.get("127.2", current_price * 1.05)
            tp2_price = extensions.get("161.8", current_price * 1.10)
            tp3_price = extensions.get("200.0", current_price * 1.15)
            
        else:  # bearish
            direction = "SHORT"
            # For bearish, we sell retracements and target extensions
            entry_price = current_price
            stop_loss = retracements.get("100.0", swing["swing_high"])  # Above swing high
            
            # Take profit levels (for shorts, extensions are lower)
            tp1_price = extensions.get("127.2", current_price * 0.95)
            tp2_price = extensions.get("161.8", current_price * 0.90)
            tp3_price = extensions.get("200.0", current_price * 0.85)
        
        # Calculate risk amount
        risk_amount = account_balance * (risk_percent / 100)
        risk_per_trade = abs(entry_price - stop_loss)
        
        # Calculate position size
        if risk_per_trade > 0:
            position_size = risk_amount / risk_per_trade
        else:
            position_size = 0
            
        # Calculate take profit levels with risk/reward ratios
        take_profits = []
        for i, (tp_price, size_pct) in enumerate([(tp1_price, "50%"), (tp2_price, "25%"), (tp3_price, "25%")], 1):
            if tp_price > 0:
                reward = abs(tp_price - entry_price)
                rr_ratio = reward / risk_per_trade if risk_per_trade > 0 else 0
                take_profits.append({
                    "price": round(tp_price, 2),
                    "size": size_pct,
                    "rr_ratio": round(rr_ratio, 1)
                })
        
        # Determine if we should enter
        should_enter = (
            position_analysis.get("in_entry_zone", False) and
            len(fib_analysis.get("confluences", [])) >= 2 and
            not position_analysis.get("near_invalidation", False)
        )
        
        return {
            "symbol": symbol,
            "direction": direction,
            "should_enter": should_enter,
            "entry_price": round(entry_price, 2),
            "stop_loss": round(stop_loss, 2),
            "risk_amount": round(risk_amount, 2),
            "risk_percent": risk_percent,
            "position_size": round(position_size, 6),
            "take_profit_1": take_profits[0] if len(take_profits) > 0 else {},
            "take_profit_2": take_profits[1] if len(take_profits) > 1 else {},
            "take_profit_3": take_profits[2] if len(take_profits) > 2 else {},
            "confluences": len(fib_analysis.get("confluences", [])),
            "position_analysis": position_analysis.get("recommendation", "No clear signal"),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def format_fibonacci_alert(self, fib_analysis: Dict) -> str:
        """
        Format Fibonacci analysis into a readable alert message
        
        Args:
            fib_analysis: Result from analyze_fibonacci()
            
        Returns: Formatted alert string
        """
        if not fib_analysis or "error" in fib_analysis:
            return f"❌ {fib_analysis.get('error', 'No Fibonacci analysis available')}"
        
        symbol = fib_analysis["symbol"]
        current_price = fib_analysis["current_price"]
        swing = fib_analysis["swing"]
        position_analysis = fib_analysis["position_analysis"]
        confluences = fib_analysis.get("confluences", [])
        
        # Header
        alert = f"📊 **FIBONACCI ANALYSIS - {symbol}**\n"
        alert += f"💰 Current Price: ${current_price:,.2f}\n"
        alert += f"📈 Swing Type: {swing['swing_type'].upper()}\n"
        alert += f"📏 Move: {swing['move_pct']:.2f}%\n\n"
        
        # Position analysis
        if position_analysis.get("in_entry_zone"):
            alert += f"🎯 **ENTRY ZONE ACTIVE**\n"
        elif position_analysis.get("at_key_level"):
            alert += f"⭐ **AT KEY LEVEL**\n"
        elif position_analysis.get("near_invalidation"):
            alert += f"🚫 **NEAR INVALIDATION**\n"
        else:
            alert += f"⏸️ **WAIT FOR BETTER ENTRY**\n"
        
        # Recommendation
        alert += f"💡 {position_analysis.get('recommendation', 'No clear signal')}\n\n"
        
        # Key Fibonacci levels
        retracements = fib_analysis["fibonacci_levels"]["retracements"]
        alert += "📉 **Key Retracement Levels:**\n"
        for level in ["38.2", "50.0", "61.8", "78.6"]:
            if level in retracements:
                price = retracements[level]
                distance = abs(current_price - price) / current_price * 100
                alert += f"   {level}%: ${price:,.2f} ({distance:.1f}% away)\n"
        
        # Confluences
        if confluences:
            alert += f"\n🔥 **Confluences Detected: {len(confluences)}**\n"
            for i, conf in enumerate(confluences[:3], 1):  # Show top 3
                alert += f"   {i}. ${conf['price']:,.2f} - {conf['confluence_count']} factors\n"
        
        # Extensions (profit targets)
        extensions = fib_analysis["fibonacci_levels"]["extensions"]
        alert += f"\n🎯 **Profit Targets:**\n"
        for level in ["127.2", "161.8", "200.0"]:
            if level in extensions:
                price = extensions[level]
                distance = abs(current_price - price) / current_price * 100
                alert += f"   {level}%: ${price:,.2f} ({distance:.1f}% away)\n"
        
        return alert

    def calculate_risk_reward(self, entry_price: float, stop_loss: float, take_profit: float) -> Dict:
        """
        Calculate risk/reward ratio and position sizing
        Returns: Dict with risk/reward metrics
        """
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)

        if risk == 0:
            return {
                "risk_reward_ratio": 0,
                "risk_amount": 0,
                "reward_amount": 0,
                "risk_percent": 0,
                "reward_percent": 0
            }

        rr_ratio = reward / risk
        risk_percent = (risk / entry_price) * 100
        reward_percent = (reward / entry_price) * 100

        return {
            "risk_reward_ratio": round(rr_ratio, 2),
            "risk_amount": round(risk, 2),
            "reward_amount": round(reward, 2),
            "risk_percent": round(risk_percent, 2),
            "reward_percent": round(reward_percent, 2),
            "is_favorable": rr_ratio >= 2.0  # Minimum 2:1 RR for good trades
        }

    def multi_timeframe_analysis(self, symbol: str) -> Dict:
        """
        Analyze trend across multiple timeframes for confirmation
        Returns: Dict with analysis for each timeframe
        """
        timeframes = {
            "15m": "15m",   # Short-term (intraday)
            "1h": "1h",     # Medium-term
            "4h": "4h",     # Longer-term
        }

        results = {}

        for name, interval in timeframes.items():
            opens, highs, lows, closes, volumes = self.fetch_historical_prices(symbol, 100, interval)

            if not closes or len(closes) < 50:
                results[name] = {"error": "Insufficient data"}
                continue

            closes_float = [float(c) for c in closes]
            highs_float = [float(h) for h in highs]
            lows_float = [float(l) for l in lows]

            # Calculate indicators
            trend_direction, _ = self.calculate_trend_direction(closes_float, highs_float, lows_float)
            trend_strength, strength_details = self.calculate_trend_strength(highs_float, lows_float, closes_float)

            results[name] = {
                "direction": trend_direction,
                "strength": float(trend_strength),
                "adx": strength_details.get("adx", 0),
                "is_strong": float(trend_strength) >= 25
            }

        # Check for alignment
        directions = [r["direction"] for r in results.values() if "direction" in r]
        aligned = len(set(directions)) == 1 if directions else False

        results["alignment"] = {
            "is_aligned": aligned,
            "consensus_direction": directions[0] if aligned and directions else "mixed"
        }

        return results

    def calculate_trend_direction(self, closes: List[float], highs: List[float], lows: List[float]) -> Tuple[str, Dict]:
        """
        Calculate trend direction using multiple proven indicators
        Returns: (trend_direction, details_dict)
        """
        if len(closes) < 50:
            return "sideways", {}

        # Advanced features: RSI, support/resistance, risk/reward hints and volume placeholder
        # RSI (14)
        rsi = self.calculate_rsi(closes, 14)
        rsi_signal = "neutral"
        if rsi > 70:
            rsi_signal = "overbought"
        elif rsi < 30:
            rsi_signal = "oversold"

        # Volume analysis placeholder (volumes not provided to this function).
        # If caller provides volumes via an attribute or external call, populate `volume_trend`.
        volume_trend = ("unknown", 1.0)

        # Bundle details that will be returned / used by caller
        extra_indicators = {
            "rsi": rsi,
            "rsi_signal": rsi_signal,
            "volume_trend": {"trend": volume_trend[0], "ratio": volume_trend[1]},
        }

        ema_9 = self.calculate_ema(closes, 9)
        ema_21 = self.calculate_ema(closes, 21)

        ema_signal = "neutral"
        if ema_9 and ema_21:
            if ema_9[-1] > ema_21[-1]:
                ema_signal = "bullish"
            elif ema_9[-1] < ema_21[-1]:
                ema_signal = "bearish"

        # 2. MACD
        macd, signal, histogram = self.calculate_macd(closes)
        macd_signal = "neutral"
        if histogram > 0:
            macd_signal = "bullish"
        elif histogram < 0:
            macd_signal = "bearish"

        # 3. ADX with +DI and -DI
        adx, plus_di, minus_di = self.calculate_adx(highs, lows, closes)
        adx_signal = "neutral"
        if plus_di > minus_di:
            adx_signal = "bullish"
        elif minus_di > plus_di:
            adx_signal = "bearish"

        # 4. Price vs EMA 50 (longer-term trend)
        ema_50 = self.calculate_ema(closes, 50)
        price_signal = "neutral"
        if ema_50:
            if closes[-1] > ema_50[-1]:
                price_signal = "bullish"
            elif closes[-1] < ema_50[-1]:
                price_signal = "bearish"

        # 5. RSI (avoid extreme overbought/oversold)
        # calculate_rsi is expected to return a single float value for the latest RSI
        rsi = self.calculate_rsi(closes)
        rsi_signal = "neutral"
        if rsi is not None:
            if rsi > 50:
                rsi_signal = "bullish"
            elif rsi < 50:
                rsi_signal = "bearish"

        # Combine signals with weighted voting
        bullish_votes = 0.0
        bearish_votes = 0.0

        # EMA crossover - weight: 2 (most important for direction)
        if ema_signal == "bullish":
            bullish_votes += 2
        elif ema_signal == "bearish":
            bearish_votes += 2

        # MACD - weight: 2 (momentum confirmation)
        if macd_signal == "bullish":
            bullish_votes += 2
        elif macd_signal == "bearish":
            bearish_votes += 2

        # ADX directional - weight: 1.5
        if adx_signal == "bullish":
            bullish_votes += 1.5
        elif adx_signal == "bearish":
            bearish_votes += 1.5

        # Price vs EMA50 - weight: 1 (context)
        if price_signal == "bullish":
            bullish_votes += 1
        elif price_signal == "bearish":
            bearish_votes += 1

        # RSI - weight: 0.5 (confirmation)
        if rsi_signal == "bullish":
            bullish_votes += 0.5
        elif rsi_signal == "bearish":
            bearish_votes += 0.5

        # Determine final trend
        total_votes = bullish_votes + bearish_votes
        if total_votes == 0:
            trend = "sideways"
        else:
            bullish_pct = (bullish_votes / total_votes) * 100
            bearish_pct = (bearish_votes / total_votes) * 100

            # Need at least 60% consensus for a trend
            if bullish_pct >= 60:
                trend = "uptrend"
            elif bearish_pct >= 60:
                trend = "downtrend"
            else:
                trend = "sideways"

        details = {
            "ema_9": round(ema_9[-1], 2) if ema_9 else 0,
            "ema_21": round(ema_21[-1], 2) if ema_21 else 0,
            "ema_signal": ema_signal,
            "macd": round(macd, 4),
            "macd_signal": round(signal, 4),
            "macd_histogram": round(histogram, 4),
            "macd_trend": macd_signal,
            "adx": round(adx, 2),
            "plus_di": round(plus_di, 2),
            "minus_di": round(minus_di, 2),
            "adx_trend": adx_signal,
            "rsi": round(rsi, 2) if rsi is not None else None,
            "rsi_signal": rsi_signal,
            "bullish_votes": bullish_votes,
            "bearish_votes": bearish_votes
        }

        return trend, details


    def analyze_trend_advanced(self, symbol: str) -> Optional[Dict]:
        """
        Advanced trend analysis with all profitability features
        Returns: Comprehensive analysis dict
        """
        print(f"\n{'='*70}")
        print(f"🔍 ADVANCED ANALYSIS: {symbol}")
        print(f"{'='*70}")

        # Fetch data with volume
        opens, highs, lows, closes, volumes = self.fetch_historical_prices(symbol, 100, "15m")

        if not closes or len(closes) < 50:
            print(f"❌ Insufficient data for {symbol}")
            return None

        # Convert to float arrays
        closes_float = [float(c) for c in closes]
        highs_float = [float(h) for h in highs]
        lows_float = [float(l) for l in lows]
        current_price = closes_float[-1]

        # 1. Basic trend analysis
        trend_direction, direction_details = self.calculate_trend_direction(closes_float, highs_float, lows_float)
        trend_strength, strength_details = self.calculate_trend_strength(highs_float, lows_float, closes_float)

        # 2. RSI analysis
        # 3. Support/Resistance levels
        sr_levels = self.find_support_resistance(highs_float, lows_float, closes_float)

        # 4. RSI for overbought/oversold
        rsi = self.calculate_rsi(closes_float)
        rsi_signal = "neutral"
        if rsi > 70:
            rsi_signal = "overbought"
        elif rsi < 30:
            rsi_signal = "oversold"

        # Define RSI status for the return data
        rsi_status = rsi_signal

        # 5. Volume trend analysis
        volume_trend, volume_ratio = self.calculate_volume_trend(volumes)

        # 6. Multi-timeframe analysis
        mtf_analysis = self.multi_timeframe_analysis(symbol)

        # 7. Fibonacci analysis (if available)
        try:
            fib_analysis = self.analyze_fibonacci(symbol, lookback=100, interval="15m")
        except:
            fib_analysis = None

        # 8. Bollinger Bands analysis
        try:
            bb_analysis = self.calculate_bollinger_bands(closes_float, period=20, multiplier=2.0)
        except:
            bb_analysis = None
        sr_levels = self.find_support_resistance(highs_float, lows_float, closes_float)

        # 4.5. Fibonacci Analysis (NEW)
        fib_analysis = self.analyze_fibonacci(symbol, lookback=100, interval="15m")

        # Get EMA levels for Fibonacci confluence
        ema_9 = self.calculate_ema(closes_float, 9)
        ema_21 = self.calculate_ema(closes_float, 21)
        ema_50 = self.calculate_ema(closes_float, 50)

        # 5. Multi-timeframe confirmation
        mtf_analysis = self.multi_timeframe_analysis(symbol)

        # 6. Calculate entry/exit levels (enhanced with Fibonacci)
        entry_price = current_price

        # Use Fibonacci levels if available, otherwise fall back to S/R
        if fib_analysis and "error" not in fib_analysis:
            fib_entry_zone = fib_analysis.get("entry_zone", {})
            fib_targets = fib_analysis.get("profit_targets", {})

            if trend_direction == "uptrend":
                # For long positions - use Fibonacci retracement for entry, extensions for targets
                stop_loss = fib_entry_zone.get("invalidation", sr_levels.get("nearest_support") or (current_price * 0.97))
                take_profit = fib_targets.get("target_2", sr_levels.get("nearest_resistance") or (current_price * 1.06))
            elif trend_direction == "downtrend":
                # For short positions
                stop_loss = fib_entry_zone.get("invalidation", sr_levels.get("nearest_resistance") or (current_price * 1.03))
                take_profit = fib_targets.get("target_2", sr_levels.get("nearest_support") or (current_price * 0.94))
            else:
                # Sideways - use tighter stops
                stop_loss = current_price * 0.98
                take_profit = current_price * 1.04
        else:
            # Fallback to traditional S/R levels
            if trend_direction == "uptrend":
                stop_loss = sr_levels.get("nearest_support") or (current_price * 0.97)
                take_profit = sr_levels.get("nearest_resistance") or (current_price * 1.06)
            elif trend_direction == "downtrend":
                stop_loss = sr_levels.get("nearest_resistance") or (current_price * 1.03)
                take_profit = sr_levels.get("nearest_support") or (current_price * 0.94)
            else:
                stop_loss = current_price * 0.98
                take_profit = current_price * 1.04

        # 7. Risk/Reward calculation
        rr_metrics = self.calculate_risk_reward(entry_price, stop_loss, take_profit)

        # 8. Generate trading signal with confidence score (ENHANCED WITH FIBONACCI AND ML)
        confidence_score = 0
        reasons = []
        warnings = []

        # Check ADX strength
        adx_val = strength_details.get("adx", 0)
        if adx_val >= 25:
            confidence_score += 30
            reasons.append(f"Strong trend (ADX: {adx_val:.1f})")
        else:
            warnings.append(f"Weak trend (ADX: {adx_val:.1f} < 25)")

        # NEW: ML Pattern Recognition Integration
        try:
            # Attempt to import the ML predictor module if available
            from ml_trading_system.ml_predictor import MLPredictor

            try:
                # Instantiate predictor and request a prediction for this symbol/timeframe
                # Using "15m" as the analysis timeframe (adjust if your method provides a different variable)
                ml_predictor = MLPredictor()
                ml_prediction = ml_predictor.predict_trade_profitability(symbol, "15m")

                # Expecting ml_prediction to be a dict with a 'confidence_score' (percentage 0-100)
                if ml_prediction and isinstance(ml_prediction, dict):
                    ml_confidence = ml_prediction.get("confidence_score") or ml_prediction.get("confidence")
                    if ml_confidence is not None:
                        try:
                            ml_confidence = float(ml_confidence)
                        except Exception:
                            ml_confidence = None

                    if ml_confidence is not None:
                        if ml_confidence > 70:
                            confidence_score += 20  # +20 points for high ML confidence
                            reasons.append(f"ML Pattern Recognition: High confidence ({ml_confidence:.1f}%)")
                        elif ml_confidence > 60:
                            confidence_score += 10  # +10 points for moderate ML confidence
                            reasons.append(f"ML Pattern Recognition: Moderate confidence ({ml_confidence:.1f}%)")
                        else:
                            warnings.append(f"ML Pattern Recognition: Low confidence ({ml_confidence:.1f}%)")
            except Exception as e:
                # ML runtime error; don't interrupt main flow
                print(f"ML prediction runtime error: {e}")
        except ImportError:
            # ML system not installed / available; continue without ML adjustments
            pass
        except Exception as e:
            # Any other import-related error; continue without ML
            print(f"ML integration error: {e}")
            pass

        # Check multi-timeframe alignment
        if mtf_analysis.get("alignment", {}).get("is_aligned", False):
            confidence_score += 25
            reasons.append("Multi-timeframe alignment")
        else:
            warnings.append("Timeframes not aligned")

        # Check RSI (avoid extremes for entries)
        if trend_direction == "uptrend" and rsi < 70:
            confidence_score += 15
            reasons.append(f"RSI not overbought ({rsi:.1f})")
        elif trend_direction == "downtrend" and rsi > 30:
            confidence_score += 15
            reasons.append(f"RSI not oversold ({rsi:.1f})")
        elif rsi > 70 or rsi < 30:
            warnings.append(f"RSI extreme ({rsi:.1f})")

        # Check volume confirmation
        if volume_trend == "increasing":
            confidence_score += 15
            reasons.append(f"Volume increasing ({volume_ratio:.2f}x)")
        elif volume_trend == "decreasing":
            warnings.append(f"Volume decreasing ({volume_ratio:.2f}x)")

        # Check risk/reward ratio
        if rr_metrics["is_favorable"]:
            confidence_score += 15
            reasons.append(f"Good R:R ({rr_metrics['risk_reward_ratio']}:1)")
        else:
            warnings.append(f"Poor R:R ({rr_metrics['risk_reward_ratio']}:1)")

        # NEW: Bollinger Bands-based confidence scoring
        if bb_analysis:
            # Bonus for squeeze detection (high probability setups)
            if bb_analysis.get("squeeze"):
                confidence_score += 20
                reasons.append("Bollinger Bands squeeze detected (high volatility expansion expected)")

            # Bonus for strong band walk (trend confirmation)
            band_walk = bb_analysis.get("band_walk", "")
            if band_walk == "upper":
                # In strong uptrend - confirm with trend direction
                if trend_direction == "uptrend":
                    confidence_score += 15
                    reasons.append("Price riding upper Bollinger Band (strong uptrend confirmation)")
                else:
                    warnings.append("Price above upper band but trend direction conflicts")
            elif band_walk == "lower":
                # In strong downtrend - confirm with trend direction
                if trend_direction == "downtrend":
                    confidence_score += 15
                    reasons.append("Price riding lower Bollinger Band (strong downtrend confirmation)")
                else:
                    warnings.append("Price below lower band but trend direction conflicts")

            # Bonus for mean reversion setups (in ranging markets)
            mean_reversion = bb_analysis.get("mean_reversion_signal", "")
            adx_val = strength_details.get("adx", 0)
            if mean_reversion and adx_val < 25:  # Only in ranging markets
                if mean_reversion == "buy":
                    confidence_score += 10
                    reasons.append("Bollinger Bands mean reversion buy setup (near lower band in range)")
                elif mean_reversion == "sell":
                    confidence_score += 10
                    reasons.append("Bollinger Bands mean reversion sell setup (near upper band in range)")

            # Bonus for breakout confirmation (with volume)
            breakout = bb_analysis.get("breakout_signal", "")
            if breakout and volume_trend == "increasing":
                if breakout == "strong_up":
                    confidence_score += 15
                    reasons.append("Bollinger Bands breakout above upper band with volume confirmation")
                elif breakout == "strong_down":
                    confidence_score += 15
                    reasons.append("Bollinger Bands breakout below lower band with volume confirmation")

            # Warning for extreme bandwidth (high volatility)
            bandwidth = bb_analysis.get("bandwidth", 0)
            if bandwidth > 5:  # High volatility
                warnings.append(f"High volatility detected (Bandwidth: {bandwidth:.2f}%) - smaller positions advised")

        # NEW: Fibonacci-based confidence scoring
        if fib_analysis and "error" not in fib_analysis:
            # Bonus for being at key Fibonacci level (38.2%, 50%, 61.8%)
            nearest_fib = fib_analysis.get("nearest_fib_level")
            if nearest_fib and nearest_fib.get("is_key_level"):
                confidence_score += 10
                reasons.append(f"At key Fibonacci level ({nearest_fib['level_name']})")

            # Bonus for being in optimal entry zone
            position_analysis = fib_analysis.get("position_analysis", {})
            if position_analysis.get("in_entry_zone"):
                confidence_score += 10
                reasons.append("Price in Fibonacci entry zone (38.2%-61.8%)")

            # Bonus for Fibonacci confluence
            confluences = fib_analysis.get("confluences", [])
            if confluences:
                # Strong confluence (3+ factors)
                strong_confluences = [c for c in confluences if c["confluence_count"] >= 3]
                if strong_confluences:
                    confidence_score += 15
                    reasons.append(f"Strong Fibonacci confluence ({strong_confluences[0]['confluence_count']} factors)")
                elif len(confluences) >= 2:
                    confidence_score += 10
                    reasons.append(f"Fibonacci confluence detected ({len(confluences)} zones)")

            # Warning if near invalidation
            if position_analysis.get("near_invalidation"):
                warnings.append("Price near Fibonacci invalidation level")

            # Check if ADX confirms trending market (Fibonacci works best in trends)
            if adx_val < 25:
                warnings.append("Fibonacci less reliable in weak trends (ADX < 25)")

            # ------------------------------
            # Add Bollinger Bands summary to reasons and enrich bb_analysis
            # ------------------------------
            if bb_analysis:
                # safe formatter for numeric values
                def _fmt(v, perc=False):
                    try:
                        f = float(v)
                        return f"{f:.2f}" + ("%" if perc else "")
                    except Exception:
                        return str(v)

                mid = bb_analysis.get("middle_band", None)
                upper = bb_analysis.get("upper_band", None)
                lower = bb_analysis.get("lower_band", None)
                bw = bb_analysis.get("bandwidth", None)
                percent_b = bb_analysis.get("percent_b", None)
                squeeze = bool(bb_analysis.get("squeeze", False))
                band_walk = bb_analysis.get("band_walk", "") or "none"
                mean_rev = bb_analysis.get("mean_reversion_signal", "") or "none"
                breakout_sig = bb_analysis.get("breakout_signal", "") or "none"

                summary = (
                    f"Bollinger Bands - Mid: {_fmt(mid)}, Upper: {_fmt(upper)}, Lower: {_fmt(lower)} | "
                    f"Bandwidth: {_fmt(bw, perc=True)} | %B: {_fmt(percent_b)} | "
                    f"Squeeze: {'Yes' if squeeze else 'No'} | Walk: {band_walk} | "
                    f"MeanReversion: {mean_rev} | Breakout: {breakout_sig}"
                )

                # Add concise summary to reasons (helps explain the score)
                reasons.append(summary)

                # Enrich bb_analysis dict so the result contains a ready-to-display summary and flags
                try:
                    bb_analysis["summary"] = summary
                    bb_analysis["flags"] = {
                        "squeeze": squeeze,
                        "band_walk": band_walk,
                        "mean_reversion": mean_rev,
                        "breakout": breakout_sig,
                    }
                    # store nicely formatted numeric snapshot
                    bb_analysis["snapshot"] = {
                        "middle_band": float(mid) if mid is not None else None,
                        "upper_band": float(upper) if upper is not None else None,
                        "lower_band": float(lower) if lower is not None else None,
                        "bandwidth_pct": float(bw) if bw is not None else None,
                        "percent_b": float(percent_b) if percent_b is not None else None,
                    }
                except Exception:
                    # If conversion fails, keep the original bb_analysis untouched beyond the summary/flags
                    pass

        # Determine final signal
        if confidence_score >= 70:
            signal_strength = "STRONG"
        elif confidence_score >= 50:
            signal_strength = "MODERATE"
        elif confidence_score >= 30:
            signal_strength = "WEAK"
        else:
            signal_strength = "NO TRADE"

        # Compile results (ENHANCED WITH FIBONACCI DATA)
        analysis = {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "current_price": current_price,

            # Trend
            "trend_direction": trend_direction,
            "trend_strength": float(trend_strength),
            "adx": adx_val,

            # Indicators
            "rsi": rsi,
            "rsi_status": rsi_status,
            "volume_trend": volume_trend,
            "volume_ratio": volume_ratio,

            # Levels
            "support_levels": sr_levels.get("support", []),
            "resistance_levels": sr_levels.get("resistance", []),
            "nearest_support": sr_levels.get("nearest_support"),
            "nearest_resistance": sr_levels.get("nearest_resistance"),

            # Fibonacci Analysis (NEW)
            "fibonacci": fib_analysis if fib_analysis and "error" not in fib_analysis else None,
            "fibonacci_available": fib_analysis is not None and "error" not in fib_analysis,

            # Multi-timeframe
            "timeframe_analysis": mtf_analysis,
            "timeframes_aligned": mtf_analysis.get("alignment", {}).get("is_aligned", False),

            # Trade setup
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_reward": rr_metrics,

            # Signal
            "signal_strength": signal_strength,
            "confidence_score": confidence_score,
            "reasons": reasons,
            "warnings": warnings,

            # Recommendation
            "action": self._get_trade_action(trend_direction, signal_strength, adx_val, rsi)
        }

        return analysis

    def _get_trade_action(self, trend: str, signal_strength: str, adx: float, rsi: float) -> str:
        """Generate specific trade action recommendation"""
        if signal_strength == "NO TRADE":
            return "⏸️ STAY OUT - Wait for better setup"

        if trend == "uptrend":
            if signal_strength == "STRONG":
                return "✅ STRONG BUY - Enter LONG position"
            elif signal_strength == "MODERATE":
                return "⚠️ MODERATE BUY - Consider small LONG position"
            else:
                return "⚠️ WEAK BUY - High risk, small position only"

        elif trend == "downtrend":
            if signal_strength == "STRONG":
                return "❌ STRONG SELL - Enter SHORT position"
            elif signal_strength == "MODERATE":
                return "⚠️ MODERATE SELL - Consider small SHORT position"
            else:
                return "⚠️ WEAK SELL - High risk, small position only"

        else:  # sideways
            return "⏸️ NO CLEAR TREND - Wait for breakout"

    def calculate_trend_strength(self, highs: List[float], lows: List[float], closes: List[float]) -> Tuple[Decimal, Dict]:
        """
        Calculate trend strength using ADX (industry standard)
        Returns: (strength 0-100, details_dict)

        ADX Interpretation:
        0-25: Weak/No trend (ranging market)
        25-50: Strong trend (good for trend trading)
        50-75: Very strong trend (excellent for trend trading)
        75-100: Extremely strong trend (may be overextended)
        """
        if len(closes) < 30:
            return Decimal("0"), {}

        # Calculate ADX
        adx, plus_di, minus_di = self.calculate_adx(highs, lows, closes, period=14)

        # Calculate additional strength indicators

        # 1. Volatility (ATR as % of price)
        atr = self.calculate_atr(highs, lows, closes, period=14)
        atr_pct = (atr[-1] / closes[-1] * 100) if atr and closes[-1] != 0 else 0

        # 2. Momentum (rate of change over 14 periods)
        if len(closes) >= 14:
            roc = ((closes[-1] - closes[-14]) / closes[-14] * 100) if closes[-14] != 0 else 0
        else:
            roc = 0

        details = {
            "adx": round(adx, 2),
            "plus_di": round(plus_di, 2),
            "minus_di": round(minus_di, 2),
            "atr_percent": round(atr_pct, 4),
            "momentum_roc": round(roc, 2),
            "strength_category": self._get_strength_category(adx)
        }

        # Use ADX as the primary strength metric (0-100 scale)
        return Decimal(str(round(adx, 2))), details

    def _get_strength_category(self, adx: float) -> str:
        """Categorize trend strength based on ADX value"""
        if adx < 25:
            return "Weak/Ranging"
        elif adx < 50:
            return "Strong Trend"
        elif adx < 75:
            return "Very Strong Trend"
        else:
            return "Extremely Strong Trend"

    def analyze_trend(self, symbol: str) -> Optional[Trend]:
        """Analyze trend for a given symbol using professional indicators"""
        opens, highs, lows, closes, volumes = self.fetch_historical_prices(symbol, 100)

        if not closes or len(closes) < 50:
            print(f"Insufficient price data for {symbol}")
            return None

        # Convert to float arrays for calculations
        closes_float = [float(c) for c in closes]
        highs_float = [float(h) for h in highs]
        lows_float = [float(l) for l in lows]

        # Calculate trend direction
        trend_direction, direction_details = self.calculate_trend_direction(closes_float, highs_float, lows_float)

        # Calculate trend strength
        trend_strength, strength_details = self.calculate_trend_strength(highs_float, lows_float, closes_float)

        # Log details for debugging
        print(f"\n{symbol} Analysis:")
        print(f"  Direction: {trend_direction}")
        print(f"  Strength: {trend_strength} ({strength_details.get('strength_category', 'N/A')})")
        print(f"  ADX: {strength_details.get('adx', 0)}, +DI: {strength_details.get('plus_di', 0)}, -DI: {strength_details.get('minus_di', 0)}")
        print(f"  EMA Signal: {direction_details.get('ema_signal', 'N/A')}")
        print(f"  MACD: {direction_details.get('macd_trend', 'N/A')}")
        print(f"  Strength: {trend_strength} ({strength_details.get('strength_category', 'N/A')})")
        print(f"  ADX: {strength_details.get('adx', 0)}, +DI: {strength_details.get('plus_di', 0)}, -DI: {strength_details.get('minus_di', 0)}")
        print(f"  EMA Signal: {direction_details.get('ema_signal', 'N/A')}")
        print(f"  MACD: {direction_details.get('macd_trend', 'N/A')}")

        return Trend(
            pair=symbol,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            calculated_at=datetime.now(timezone.utc)
        )

    def save_trend(self, trend: Trend) -> bool:
        """Save trend analysis to database"""
        try:
            trend_data = {
                "pair": trend.pair,
                "trend_direction": trend.trend_direction,
                "trend_strength": float(trend.trend_strength),
                "calculated_at": trend.calculated_at.isoformat()
            }

            result = supabase.table("trends").insert(trend_data).execute()
            return bool(result.data)
        except Exception as e:
            print(f"Error saving trend: {e}")
            return False

    def get_unique_pairs(self) -> List[str]:
        """Get unique trading pairs from price levels"""
        try:
            result = supabase.table("price_levels").select("pair").execute()
            pairs = list(set([row["pair"] for row in result.data]))
            return pairs
        except Exception as e:
            print(f"Error fetching pairs: {e}")
            return []

    def run_trend_analysis(self) -> int:
        """Run trend analysis for all unique pairs"""
        pairs = self.get_unique_pairs()
        analyzed_count = 0

        print(f"Analyzing trends for {len(pairs)} pairs...")

        for pair in pairs:
            print(f"\n{'='*60}")
            print(f"Analyzing trend for {pair}...")
            print(f"{'='*60}")
            trend = self.analyze_trend(pair)

            if trend:
                if self.save_trend(trend):
                    print(f"✅ Trend analysis saved for {pair}: {trend.trend_direction} (strength: {trend.trend_strength})")
                    analyzed_count += 1
                else:
                    print(f"❌ Failed to save trend for {pair}")
            else:
                print(f"❌ Failed to analyze trend for {pair}")

        return analyzed_count
