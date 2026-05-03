"""The UI render on browser."""

from fastapi import APIRouter

from .tags import APITag

ui_router = APIRouter(prefix="/ui", tags=[APITag.UI])
