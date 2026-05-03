"""
tableauknockoff - FastAPI Application
"""

from collections.abc import Awaitable, Callable
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, Response
from fastui import prebuilt_html

from web import OPENAPI_TAGS, APITag, api_router, ui_router

app = FastAPI(
    title="tableauknockoff",
    description="A FastAPI application",
    version="0.1.0",
    openapi_tags=OPENAPI_TAGS,
)

# Add all URL routes
app.include_router(api_router)
app.include_router(ui_router)


# Time check Middleware
@app.middleware("http")
async def add_process_time_header(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time

    response.headers["X-Process-Time"] = str(process_time)
    return response


# GZip Compression Middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=[APITag.METADATA])
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/{_:path}", tags=[APITag.UI_HANDLER])
async def path_catcher(_: str):
    """Catches all paths and lets FastUI handle it."""
    return HTMLResponse(prebuilt_html(api_root_url="/ui"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
