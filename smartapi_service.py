"""Server-side Angel One SmartAPI client and conservative technical signal logic."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any


class SmartAPIUnavailable(RuntimeError):
    """SmartAPI credentials or its optional package are unavailable."""


class SmartMarketService:
    def __init__(self) -> None:
        self.api_key = os.getenv("SMART_API_KEY")
        self.client_code = os.getenv("SMART_CLIENT_CODE")
        self.pin = os.getenv("SMART_PIN")
        self.totp_secret = os.getenv("SMART_TOTP_SECRET")
        self.client: Any | None = None

    @property
    def configured(self) -> bool:
        return all((self.api_key, self.client_code, self.pin, self.totp_secret))

    def _client(self) -> Any:
        if not self.configured:
            raise SmartAPIUnavailable("SmartAPI is not configured. Add SMART_API_KEY, SMART_CLIENT_CODE, SMART_PIN, and SMART_TOTP_SECRET to the server environment.")
        if self.client is None:
            try:
                import pyotp
                from SmartApi import SmartConnect
            except ImportError as error:
                raise SmartAPIUnavailable("SmartAPI dependencies are missing. Run: pip install -r requirements.txt") from error
            client = SmartConnect(self.api_key)
            session = client.generateSession(self.client_code, self.pin, pyotp.TOTP(self.totp_secret).now())
            if not session.get("status"):
                raise SmartAPIUnavailable(session.get("message", "SmartAPI authentication failed."))
            self.client = client
        return self.client

    def search(self, query: str, exchange: str) -> list[dict[str, str]]:
        response = self._client().searchScrip(exchange, query)
        if not response.get("status"):
            raise SmartAPIUnavailable(response.get("message", "SmartAPI search failed."))
        return [
            {"exchange": item["exchange"], "tradingsymbol": item["tradingsymbol"], "symboltoken": item["symboltoken"]}
            for item in response.get("data") or []
        ]

    def quote(self, exchange: str, symboltoken: str, tradingsymbol: str) -> dict[str, float | str]:
        data = self._client().ltpData(exchange, tradingsymbol, symboltoken)
        if not data.get("status"):
            raise SmartAPIUnavailable(data.get("message", "SmartAPI quote request failed."))
        quote = data["data"]
        return {"exchange": exchange, "symboltoken": symboltoken, "tradingsymbol": tradingsymbol, "ltp": float(quote["ltp"]), "close": float(quote.get("close", 0)), "timestamp": datetime.now().isoformat(timespec="seconds")}

    def positions(self) -> list[dict[str, Any]]:
        response = self._client().position()
        if not response.get("status"):
            raise SmartAPIUnavailable(response.get("message", "SmartAPI positions request failed."))
        return response.get("data") or []

    def prediction(self, exchange: str, symboltoken: str, tradingsymbol: str, side: str, entry_price: float | None) -> dict[str, Any]:
        client = self._client()
        quote = self.quote(exchange, symboltoken, tradingsymbol)
        end = datetime.now()
        candles = client.getCandleData({"exchange": exchange, "symboltoken": symboltoken, "interval": "FIVE_MINUTE", "fromdate": (end - timedelta(days=7)).strftime("%Y-%m-%d %H:%M"), "todate": end.strftime("%Y-%m-%d %H:%M")})
        rows = candles.get("data") or []
        closes = [float(row[4]) for row in rows if len(row) > 4]
        if len(closes) < 30:
            raise SmartAPIUnavailable("Not enough recent candle data to calculate a signal.")
        fast = sum(closes[-10:]) / 10
        slow = sum(closes[-30:]) / 30
        momentum = (closes[-1] - closes[-15]) / closes[-15] * 100 if closes[-15] else 0
        direction = "bullish" if fast > slow and momentum > 0 else "bearish" if fast < slow and momentum < 0 else "neutral"
        confidence = min(78, round(50 + min(20, abs((fast - slow) / slow * 100) * 8) + min(8, abs(momentum))))
        current = float(quote["ltp"])
        position_return = None
        if entry_price and entry_price > 0:
            raw_return = (current - entry_price) / entry_price * 100
            position_return = raw_return if side.upper() == "BUY" else -raw_return
        return {"quote": quote, "signal": direction, "confidence": confidence, "indicators": {"ema_10_proxy": round(fast, 2), "ema_30_proxy": round(slow, 2), "five_minute_momentum_percent": round(momentum, 2)}, "position": {"side": side.upper(), "entry_price": entry_price, "unrealized_return_percent": round(position_return, 2) if position_return is not None else None}, "disclaimer": "Educational technical signal based on recent price data, not investment advice or a guarantee of future performance."}
