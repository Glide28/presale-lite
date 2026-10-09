"""Точка входа ASGI: uvicorn app.main:app"""
from app.api import create_app

app = create_app()
