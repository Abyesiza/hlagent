"""WebSocket session manager (stub for streaming agent / tool traces)."""


class WebSocketManager:
    async def connect(self, session_id: str) -> None:
        ...

    async def disconnect(self, session_id: str) -> None:
        ...
