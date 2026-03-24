"""Vercel FastAPI entrypoint.

Vercel looks for an ASGI app in api/*.py. We re-export the main app here.
"""

from super_agent.app.main import app

