from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from decimal import Decimal

class PriceLevel(BaseModel):
    id: Optional[str] = None
    pair: str = Field(..., description="Trading pair e.g., 'BTCUSDT'")
    target_price: Decimal = Field(..., description="Target price to monitor")
    trigger_direction: str = Field(default="above", description="'above' or 'below' - trigger when price goes above or below target")
    is_active: bool = Field(default=True, description="Whether this alert is active")
    trigger_type: str = Field(default="one_time", description="'one_time' or 'continuous'")
    created_at: Optional[datetime] = None
    last_triggered_at: Optional[datetime] = None

class Alert(BaseModel):
    id: Optional[str] = None
    price_level_id: str = Field(..., description="Reference to price level")
    pair: str = Field(..., description="Trading pair")
    triggered_price: Decimal = Field(..., description="Price when alert was triggered")
    target_price: Decimal = Field(..., description="Original target price")
    trigger_direction: str = Field(..., description="'above' or 'below' - direction that triggered the alert")
    trigger_type: str = Field(..., description="'one_time' or 'continuous'")
    triggered_at: Optional[datetime] = None
    notified: bool = Field(default=False, description="Whether notification was sent")

class Trend(BaseModel):
    id: Optional[str] = None
    pair: str = Field(..., description="Trading pair")
    trend_direction: str = Field(..., description="'uptrend', 'downtrend', or 'sideways'")
    trend_strength: Decimal = Field(..., description="Trend strength 0-100")
    calculated_at: Optional[datetime] = None

class PriceData(BaseModel):
    symbol: str
    price: Decimal
    timestamp: datetime
