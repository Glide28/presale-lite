"""Прикладной слой: сценарии работы с пресейлом (не зависит от FastAPI)."""
from __future__ import annotations

import logging
from typing import Iterator

from app import documents, estimator, report
from app.config import Settings
from app.llm import LLM
from app.storage import NotFound, Storage

log = logging.getLogger("presale.service")


class ValidationError(ValueError):
    pass


class PresaleService:
    def __init__(self, storage: Storage, llm: LLM, settings: Settings) -> None:
        self.storage = storage
        self.llm = llm
        self.settings = settings

    def create(self, title: str, customer: str | None, description: str,
               outputs: list[str]) -> dict:
        title = (title or "").strip()
        if len(title) < 3:
            raise ValidationError("Название должно быть не короче 3 символов")
        bad = [o for o in outputs if o not in estimator.OUTPUT_SECTIONS]
        if bad:
            raise ValidationError(f"Неизвестные результаты: {', '.join(bad)}")
        presale = self.storage.create_presale(title, customer, description or "", outputs)
        log.info("presale created id=%s", presale["id"])
        return presale

    def upload_document(self, pid: str, filename: str, content: bytes) -> dict:
        try:
            text = documents.extract_text(filename, content, self.settings.max_upload_bytes)
        except documents.DocumentError as exc:
            raise ValidationError(str(exc)) from exc
        return self.storage.add_document(pid, filename, text)

    def set_rates(self, pid: str, rates: list[dict]) -> None:
        clean = []
        for item in rates:
            role = str(item.get("role", "")).strip()
            try:
                rate = float(item.get("hourly_rate", 0))
            except (TypeError, ValueError):
                raise ValidationError("Ставка должна быть числом") from None
            if not role or rate <= 0:
                raise ValidationError("Для каждой ставки нужны роль и значение больше нуля")
            clean.append({"role": role, "hourly_rate": rate})
        self.storage.set_rates(pid, clean)

    def questions(self, pid: str) -> list[str]:
        presale = self.storage.get_presale(pid)
        docs = self.storage.list_documents(pid)
        text = presale["description"] + "\n" + "\n".join(d["text"][:3000] for d in docs)
        return estimator.generate_questions(self.llm, text)

    def save_answers(self, pid: str, qa: list[tuple[str, str]]) -> None:
        self.storage.set_answers(pid, qa)

    def run_estimate(self, pid: str) -> dict:
        presale = self.storage.get_presale(pid)
        docs = [(d["filename"], d["text"]) for d in self.storage.list_documents(pid)]
        if not docs and not presale["description"].strip():
            raise ValidationError("Добавьте описание проекта или загрузите документ")
        prompt = estimator.build_user_prompt(
            presale["title"], presale["customer"], presale["description"], docs,
            self.storage.list_answers(pid), presale["outputs"], presale["rates"],
        )
        result = estimator.estimate(self.llm, prompt, presale["rates"])
        result["report_markdown"] = report.render_markdown(presale, result)
        self.storage.save_result(pid, result)
        log.info("estimate done id=%s hours=%s", pid, result["effort_budget"]["total_hours"])
        return result

    def _chat_context(self, pid: str) -> tuple[str, list[dict]]:
        presale = self.storage.get_presale(pid)
        context = f"Проект: {presale['title']}\nЗаказчик: {presale['customer'] or 'не указан'}\n{presale['description']}"
        for doc in self.storage.list_documents(pid):
            context += f"\n\nДокумент «{doc['filename']}»:\n{doc['text'][:6000]}"
        qa = [(q, a) for q, a in self.storage.list_answers(pid) if a]
        if qa:
            context += "\n\nОтветы на вопросы преданализа:\n" + "\n".join(f"- {q}: {a}" for q, a in qa)
        if presale["result"]:
            context += "\n\n" + report.render_markdown(presale, presale["result"])
        return context, self.storage.list_messages(pid)

    @staticmethod
    def _clean_message(message: str) -> str:
        message = (message or "").strip()
        if not message:
            raise ValidationError("Пустое сообщение")
        return message

    def chat(self, pid: str, message: str) -> str:
        message = self._clean_message(message)
        context, history = self._chat_context(pid)
        answer = estimator.chat_answer(self.llm, context, history, message)
        self.storage.add_message(pid, "user", message)
        self.storage.add_message(pid, "assistant", answer)
        return answer

    def chat_stream(self, pid: str, message: str) -> Iterator[str]:
        """Проверки выполняются сразу; ответ отдаётся кусками, в историю попадает после завершения."""
        message = self._clean_message(message)
        context, history = self._chat_context(pid)

        def generate() -> Iterator[str]:
            parts: list[str] = []
            try:
                for chunk in estimator.chat_stream(self.llm, context, history, message):
                    parts.append(chunk)
                    yield chunk
            except estimator.EstimationError as exc:
                log.error("chat stream failed: %s", exc)
                yield f"\n\n[Ошибка модели: {exc}]"
                return
            self.storage.add_message(pid, "user", message)
            self.storage.add_message(pid, "assistant", "".join(parts))

        return generate()


__all__ = ["PresaleService", "ValidationError", "NotFound"]
