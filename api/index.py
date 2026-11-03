"""Vercel entrypoint bridging to the SonicFusion 2 FastAPI application."""

from backend.app.main import app

__all__ = ["app"]

