import base64
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from backend.config import (
    FRONTEND_DIST_DIR,
    SQLITE_PATH,
    get_provider,
    provider_supports_local_vision_uploads,
)
from backend.database import clear_history, get_history, init_db, save_message
from backend.graph import claimsight_agent

FRONTEND_DIST_PATH = Path(FRONTEND_DIST_DIR)
FRONTEND_INDEX_PATH = FRONTEND_DIST_PATH / "index.html"
RESERVED_PATH_PREFIXES = {"chat", "docs", "health", "history", "openapi.json", "redoc", "reset"}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="ClaimSight AI API",
    description="Multimodal auto-insurance claims triage assistant",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _frontend_build_missing_response() -> HTMLResponse:
    return HTMLResponse(
        (
            "<h1>Frontend build not found</h1>"
            "<p>Run <code>cd frontend && npm install && npm run build</code>, "
            "then restart the backend.</p>"
        ),
        status_code=503,
    )


def _resolve_frontend_file(path: str) -> Optional[Path]:
    if not path:
        return None

    candidate = (FRONTEND_DIST_PATH / path).resolve()
    try:
        candidate.relative_to(FRONTEND_DIST_PATH.resolve())
    except ValueError:
        return None

    if candidate.is_file():
        return candidate
    return None


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "provider": get_provider(),
        "sqlite_path": SQLITE_PATH,
        "frontend_built": FRONTEND_INDEX_PATH.exists(),
    }


@app.post("/chat")
async def chat(
    session_id: str = Form(...),
    query: str = Form(""),
    image: Optional[UploadFile] = File(None),
):
    query = query.strip()
    image_base64: Optional[str] = None
    image_data_url: Optional[str] = None
    image_mime_type: Optional[str] = None

    if not query and image is None:
        raise HTTPException(status_code=400, detail="Provide a text query, an image, or both.")

    if image is not None:
        image_mime_type = image.content_type or ""
        if not image_mime_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
        if not provider_supports_local_vision_uploads():
            raise HTTPException(
                status_code=400,
                detail="The configured provider does not support local image uploads for vision.",
            )
        try:
            image_bytes = await image.read()
            if get_provider() == "groq" and len(image_bytes) > 4 * 1024 * 1024:
                raise HTTPException(
                    status_code=400,
                    detail="Groq base64 image uploads must be 4MB or smaller. Resize or compress the image and try again.",
                )
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            image_data_url = f"data:{image_mime_type};base64,{image_base64}"
        except Exception as exc:
            if isinstance(exc, HTTPException):
                raise
            raise HTTPException(status_code=500, detail=f"Failed to read the uploaded image: {exc}") from exc

    try:
        conversation_history = get_history(session_id)
    except Exception:
        conversation_history = []

    original_query = query or "Please analyze the uploaded vehicle damage and explain likely coverage, documentation, and next steps."

    inputs = {
        "original_query": original_query,
        "image_data": image_base64,
        "image_mime_type": image_mime_type,
        "conversation_history": conversation_history,
        "session_id": session_id,
        "retry_count": 0,
        "pipeline_route": ["Load Conversation History"],
    }

    try:
        output = claimsight_agent.invoke(inputs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {exc}") from exc

    final_response = output.get("final_response", "Sorry, something went wrong.")
    image_analysis = output.get("image_analysis")
    pipeline_route = output.get("pipeline_route", [])
    response_mode = output.get("response_mode", "unknown")

    user_message = query or "Uploaded a vehicle damage photo for analysis."

    try:
        save_message(
            session_id=session_id,
            role="user",
            content=user_message,
            image_analysis=image_analysis,
            image_data_url=image_data_url,
        )
        save_message(
            session_id=session_id,
            role="assistant",
            content=final_response,
            pipeline_route=pipeline_route,
            response_mode=response_mode,
        )
    except Exception:
        pass

    return {
        "response": final_response,
        "image_analysis": image_analysis,
        "pipeline_route": pipeline_route,
        "response_mode": response_mode,
    }


@app.get("/history/{session_id}")
def get_session_history(session_id: str) -> dict:
    try:
        return {"history": get_history(session_id)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error retrieving history: {exc}") from exc


@app.post("/reset/{session_id}")
def reset_session(session_id: str) -> dict:
    try:
        clear_history(session_id)
        return {"status": "success", "message": f"Session {session_id} history cleared."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error clearing history: {exc}") from exc


@app.get("/", include_in_schema=False)
def serve_frontend_root():
    if not FRONTEND_INDEX_PATH.exists():
        return _frontend_build_missing_response()
    return FileResponse(FRONTEND_INDEX_PATH)


@app.get("/{full_path:path}", include_in_schema=False)
def serve_frontend_app(full_path: str):
    if not FRONTEND_INDEX_PATH.exists():
        return _frontend_build_missing_response()

    first_segment = full_path.split("/", 1)[0]
    if first_segment in RESERVED_PATH_PREFIXES:
        raise HTTPException(status_code=404, detail="Not found")

    file_path = _resolve_frontend_file(full_path)
    if file_path is not None:
        return FileResponse(file_path)

    return FileResponse(FRONTEND_INDEX_PATH)
