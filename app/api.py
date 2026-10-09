"""HTTP API (FastAPI). Тонкий слой над PresaleService."""
from __future__ import annotations

import logging
import secrets

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from app import report
from app.config import Settings
from app.estimator import EstimationError
from app.llm import build_llm
from app.service import NotFound, PresaleService, ValidationError
from app.storage import Storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("presale.api")


class PresaleCreate(BaseModel):
    title: str = Field(min_length=3, max_length=255)
    customer: str | None = Field(default=None, max_length=255)
    description: str = ""
    outputs: list[str] = Field(default_factory=list)


class Rate(BaseModel):
    role: str
    hourly_rate: float = Field(gt=0)


class AnswersIn(BaseModel):
    items: list[dict]  # [{"question": "...", "answer": "..."}]


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


def create_app(settings: Settings | None = None, service: PresaleService | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    if service is None:
        service = PresaleService(Storage(settings.db_path), build_llm(settings), settings)
    app = FastAPI(title="Presale Lite", version="1.0.0",
                  description="Упрощённый AI-ассистент для пресейла IT-проектов")

    def auth(authorization: str | None = Header(default=None)) -> None:
        expected = f"Bearer {settings.api_token}"
        if not authorization or not secrets.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="Неверный или отсутствующий токен")

    protected = [Depends(auth)]

    @app.exception_handler(ValidationError)
    async def _validation(_, exc: ValidationError):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(NotFound)
    async def _not_found(_, exc: NotFound):
        return JSONResponse(status_code=404, content={"detail": "Пресейл не найден"})

    @app.exception_handler(EstimationError)
    async def _estimation(_, exc: EstimationError):
        log.error("estimation failed: %s", exc)
        return JSONResponse(status_code=502, content={"detail": f"Ошибка оценки: {exc}"})

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "llm": service.llm.name, "model": service.llm.model}

    @app.post("/presales", status_code=201, dependencies=protected)
    def create_presale(data: PresaleCreate) -> dict:
        return service.create(data.title, data.customer, data.description, data.outputs)

    @app.get("/presales", dependencies=protected)
    def list_presales() -> list[dict]:
        return [{k: v for k, v in p.items() if k != "result"} for p in service.storage.list_presales()]

    @app.get("/presales/{pid}", dependencies=protected)
    def get_presale(pid: str) -> dict:
        presale = service.storage.get_presale(pid)
        presale["documents"] = [{"id": d["id"], "filename": d["filename"]}
                                for d in service.storage.list_documents(pid)]
        presale["messages"] = service.storage.list_messages(pid)
        return presale

    @app.delete("/presales/{pid}", status_code=204, dependencies=protected)
    def delete_presale(pid: str) -> None:
        service.storage.delete_presale(pid)

    @app.post("/presales/{pid}/documents", status_code=201, dependencies=protected)
    async def upload(pid: str, file: UploadFile = File(...)) -> dict:
        content = await file.read(settings.max_upload_bytes + 1)
        return service.upload_document(pid, file.filename or "document", content)

    @app.put("/presales/{pid}/rates", status_code=204, dependencies=protected)
    def set_rates(pid: str, rates: list[Rate]) -> None:
        service.set_rates(pid, [r.model_dump() for r in rates])

    @app.post("/presales/{pid}/questions", dependencies=protected)
    def questions(pid: str) -> dict:
        return {"questions": service.questions(pid)}

    @app.put("/presales/{pid}/answers", status_code=204, dependencies=protected)
    def answers(pid: str, data: AnswersIn) -> None:
        service.save_answers(pid, [(str(i.get("question", "")), str(i.get("answer", "")))
                                   for i in data.items])

    @app.post("/presales/{pid}/estimate", dependencies=protected)
    def estimate(pid: str) -> dict:
        return service.run_estimate(pid)

    @app.get("/presales/{pid}/report", response_class=PlainTextResponse, dependencies=protected)
    def report_md(pid: str) -> str:
        presale = service.storage.get_presale(pid)
        if not presale["result"]:
            raise HTTPException(status_code=409, detail="Оценка ещё не выполнена")
        return report.render_markdown(presale, presale["result"])

    @app.post("/presales/{pid}/chat", dependencies=protected)
    def chat(pid: str, data: ChatIn) -> dict:
        return {"answer": service.chat(pid, data.message)}

    @app.post("/presales/{pid}/chat/stream", dependencies=protected)
    def chat_stream(pid: str, data: ChatIn) -> StreamingResponse:
        stream = service.chat_stream(pid, data.message)  # ошибки валидации поднимаются до начала потока
        return StreamingResponse(stream, media_type="text/plain; charset=utf-8")

    return app

