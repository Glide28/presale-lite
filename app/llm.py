"""Слой доступа к LLM.

Два провайдера с общим интерфейсом:
* MockLLM — детерминированный офлайн-режим (демо без ключа, тесты);
* OpenAICompatibleLLM — любой API, совместимый с OpenAI Chat Completions
  (OpenAI, OpenRouter, Gemini через OpenAI-эндпоинт и др.), включая потоковый вывод.
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Iterator, Protocol

from app.config import Settings

log = logging.getLogger("presale.llm")


class LLMError(RuntimeError):
    """Ошибка обращения к LLM."""


class LLMAccessError(LLMError):
    """Модель недоступна (сеть, ключ, лимиты, таймаут): повторять запрос бессмысленно."""


class LLM(Protocol):
    name: str
    model: str

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str: ...

    def stream(self, system: str, user: str) -> Iterator[str]: ...


class OpenAICompatibleLLM:
    name = "openai"

    def __init__(self, settings: Settings) -> None:
        if not settings.llm_api_key:
            raise LLMError("Не задан LLM_API_KEY")
        self._s = settings
        self.model = settings.llm_model
        self._reachable = False

    # --- доступность ---
    def _precheck(self) -> None:
        """Быстрая проверка TCP-соединения: вместо многоминутного зависания даёт понятную ошибку.
        Пропускается, если в системе настроен прокси (тогда прямое подключение не показательно)."""
        if self._reachable:
            return
        if urllib.request.getproxies().get("https") or urllib.request.getproxies().get("http"):
            self._reachable = True
            return
        parsed = urllib.parse.urlparse(self._s.llm_base_url)
        host = parsed.hostname or ""
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            with socket.create_connection((host, port), timeout=self._s.llm_connect_timeout):
                self._reachable = True
        except OSError as exc:
            raise LLMAccessError(
                f"LLM недоступен: нет соединения с {host}:{port} за {self._s.llm_connect_timeout} с ({exc}). "
                "Расширение VPN в браузере не работает для Python: включите системный VPN/прокси."
            ) from exc

    # --- таймаут ---
    def _timeout_error(self) -> "LLMAccessError":
        hints = []
        if self._s.llm_reasoning_effort != "none":
            hints.append("LLM_REASONING_EFFORT=none")
        if not self._s.llm_provider_sort:
            hints.append("LLM_PROVIDER_SORT=throughput")
        hints.append(f"LLM_TIMEOUT больше {self._s.llm_timeout}")
        return LLMAccessError(
            f"LLM не ответил за {self._s.llm_timeout} с (модель генерирует ответ слишком долго). "
            f"Попробуйте в .env: {', '.join(hints)}."
        )

    @staticmethod
    def _is_timeout(exc: BaseException) -> bool:
        return isinstance(exc, TimeoutError) or isinstance(getattr(exc, "reason", None), TimeoutError)

    # --- запрос ---
    def _payload(self, system: str, user: str, *, json_mode: bool, stream: bool) -> dict:
        payload: dict = {
            "model": self._s.llm_model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if self._s.llm_temperature is not None:
            payload["temperature"] = self._s.llm_temperature
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self._s.llm_reasoning_effort:
            payload["reasoning"] = {"effort": self._s.llm_reasoning_effort}
        if self._s.llm_provider_sort:
            payload["provider"] = {"sort": self._s.llm_provider_sort}
        if self._s.llm_max_tokens:
            payload["max_tokens"] = self._s.llm_max_tokens
        if stream:
            payload["stream"] = True
        return payload

    def _open(self, payload: dict):
        self._precheck()
        request = urllib.request.Request(
            f"{self._s.llm_base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self._s.llm_api_key}"},
            method="POST",
        )
        return urllib.request.urlopen(request, timeout=self._s.llm_timeout)

    def _post(self, payload: dict):
        """Отправляет запрос; при HTTP 400 один раз повторяет без необязательных параметров
        (часть моделей не принимает temperature или response_format)."""
        try:
            return self._open(payload)
        except urllib.error.HTTPError as exc:
            optional = ("response_format", "temperature", "reasoning", "provider")
            if exc.code == 400 and any(k in payload for k in optional):
                lean = {k: v for k, v in payload.items() if k not in optional}
                try:
                    return self._open(lean)
                except urllib.error.HTTPError as retry_exc:
                    raise self._http_error(retry_exc) from retry_exc
                except (urllib.error.URLError, TimeoutError) as retry_exc:
                    raise self._network_error(retry_exc) from retry_exc
            raise self._http_error(exc) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise self._network_error(exc) from exc

    def _network_error(self, exc: BaseException) -> "LLMAccessError":
        return self._timeout_error() if self._is_timeout(exc) else LLMAccessError(f"LLM недоступен: {exc}")

    @staticmethod
    def _http_error(exc: urllib.error.HTTPError) -> LLMError:
        try:
            body = exc.read().decode("utf-8", errors="replace")[:300]
        except Exception:  # noqa: BLE001
            body = ""
        hints = {401: "неверный ключ", 402: "недостаточно средств", 403: "доступ запрещён",
                 404: "неверный адрес или модель", 429: "превышен лимит запросов"}
        hint = f" ({hints[exc.code]})" if exc.code in hints else ""
        return LLMAccessError(f"LLM вернул HTTP {exc.code}{hint}: {body}")

    # --- интерфейс ---
    def _complete_once(self, system: str, user: str, json_mode: bool) -> str:
        started = time.time()
        with self._post(self._payload(system, user, json_mode=json_mode, stream=False)) as resp:
            try:
                data = json.loads(resp.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"]
            except TimeoutError as exc:
                raise self._timeout_error() from exc
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                raise LLMError("Некорректный ответ LLM") from exc
        usage = data.get("usage") or {}
        log.info("llm ответ за %.1f с, токены: вход=%s выход=%s", time.time() - started,
                 usage.get("prompt_tokens"), usage.get("completion_tokens"))
        if not content or not str(content).strip():
            raise LLMError("LLM вернул пустой ответ")
        return str(content)

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        """Запрос с общим дедлайном: таймаут сокета ограничивает только паузы между байтами,
        а модель может «думать» дольше, поэтому ждём не дольше llm_timeout в сумме."""
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(self._complete_once, system, user, json_mode)
        try:
            return future.result(timeout=self._s.llm_timeout)
        except concurrent.futures.TimeoutError:
            raise self._timeout_error() from None
        finally:
            pool.shutdown(wait=False)

    def stream(self, system: str, user: str) -> Iterator[str]:
        with self._post(self._payload(system, user, json_mode=False, stream=True)) as resp:
            got_any = False
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0].get("delta", {}).get("content")
                except (KeyError, IndexError, TypeError, json.JSONDecodeError):
                    continue
                if delta:
                    got_any = True
                    yield delta
            if not got_any:
                raise LLMError("LLM вернул пустой ответ")


# --- Mock -------------------------------------------------------------------

_KEYWORD_TASKS = [
    # (ключевые слова, название, описание, {роль: часы})
    (("интеграц", "api", "1с", "sap", "erp"), "Интеграции с внешними системами",
     "Разработка адаптеров и обмена данными с внешними системами.",
     {"Backend Developer": 120, "Analyst": 40, "QA Engineer": 40}),
    (("безопасн", "персональн", "ролев", "аудит", "152"), "Безопасность и ролевая модель",
     "Аутентификация, роли, аудит действий, защита персональных данных.",
     {"Backend Developer": 80, "Architect": 24, "QA Engineer": 24}),
    (("отчёт", "отчет", "аналитик", "дашборд", "статистик"), "Отчётность и аналитика",
     "Формирование отчётов и дашбордов по ключевым показателям.",
     {"Backend Developer": 60, "Frontend Developer": 80, "Analyst": 24}),
    (("on-premise", "on premise", "внутри контура", "закрыт"), "Развёртывание в закрытом контуре",
     "Контейнеризация, сборка дистрибутива, инструкции по установке on-premise.",
     {"DevOps Engineer": 80, "Architect": 16}),
]

_BASE_TASKS = [
    ("Анализ и проектирование", "Уточнение требований, проектная документация.",
     {"Analyst": 80, "Architect": 40, "Project Manager": 40}),
    ("Основной функционал (backend)", "Реализация серверной логики и хранения данных.",
     {"Backend Developer": 200, "QA Engineer": 60}),
    ("Пользовательский интерфейс", "Дизайн и реализация клиентской части.",
     {"UX/UI Designer": 40, "Frontend Developer": 160, "QA Engineer": 40}),
    ("Тестирование и приёмка", "Системное тестирование, исправление дефектов, приёмка.",
     {"QA Engineer": 80, "Project Manager": 24}),
]


class MockLLM:
    """Правила вместо модели: возвращает валидный JSON в той же схеме."""

    name = "mock"
    model = "mock"

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        if "CHAT" in system:
            return self._chat_reply(user)
        if "QUESTIONS" in system:
            return json.dumps({"questions": [
                "Какие системы нужно интегрировать и по каким протоколам?",
                "Каковы требования к размещению и безопасности данных?",
                "Какая ожидаемая нагрузка и количество пользователей?",
            ]}, ensure_ascii=False)
        return json.dumps(self._estimate(user.lower()), ensure_ascii=False)

    def stream(self, system: str, user: str) -> Iterator[str]:
        words = self.complete(system, user).split(" ")
        for i, word in enumerate(words):
            yield word if i == len(words) - 1 else word + " "

    _CHAT_TOPICS = (
        (("риск",), "Риски"),
        (("архитектур", "стек", "монолит", "микросервис"), "Архитектура"),
        (("бюджет", "стоим", "цен", "сколько", "деньг", "гарант", "поддержк", "месяц", "расход"), "Бюджет"),
        (("требован", "функци", "нефункц"), "Требования"),
        (("задач", "час", "трудозатрат", "роль", "роли", "команд", "декомпоз"), "Декомпозиция и трудозатраты"),
    )

    def _chat_reply(self, user: str) -> str:
        """Офлайн-ответ: отбирает нужный раздел отчёта по ключевым словам вопроса."""
        context, _, _ = user.partition("История диалога:")
        question = user.rsplit("Вопрос:", 1)[-1].strip().lower()
        sections: dict[str, list[str]] = {}
        current = None
        for line in context.splitlines():
            if line.startswith("## "):
                current = line[3:].strip()
                sections[current] = []
            elif current is not None and line.strip():
                sections[current].append(line)
        if not sections:
            return "[mock] Оценка ещё не рассчитана. Откройте вкладку «Результат» и нажмите «Рассчитать пресейл»."
        for keywords, section in self._CHAT_TOPICS:
            if any(k in question for k in keywords) and sections.get(section):
                return f"[mock] {section}:\n" + "\n".join(sections[section])
        topics = ", ".join(name for _, name in self._CHAT_TOPICS)
        return (f"[mock] Уточните вопрос: я могу рассказать про разделы — {topics}. "
                "Для развёрнутых ответов подключите LLM (LLM_PROVIDER=openai).")

    def _estimate(self, text: str) -> dict:
        tasks = [{"name": n, "description": d,
                  "estimates": [{"role": r, "hours": h} for r, h in est.items()]}
                 for n, d, est in _BASE_TASKS]
        risks = [{"risk": "Требования могут измениться в ходе проекта", "impact": "medium",
                  "mitigation": "Поэтапная приёмка и фиксация изменений через change request"}]
        fr = [{"code": "FR-001", "title": "Создание и ведение заявки на пресейл", "priority": "must"},
              {"code": "FR-002", "title": "Загрузка документов заказчика", "priority": "must"}]
        nfr = [{"code": "NFR-001", "title": "Производительность",
                "description": "Ответ на типовые запросы не более 3 секунд"}]
        for keywords, name, desc, est in _KEYWORD_TASKS:
            if any(k in text for k in keywords):
                tasks.append({"name": name, "description": desc,
                              "estimates": [{"role": r, "hours": h} for r, h in est.items()]})
                fr.append({"code": f"FR-{len(fr) + 1:03d}", "title": name, "priority": "should"})
        if any(k in text for k in ("безопасн", "персональн", "152")):
            nfr.append({"code": f"NFR-{len(nfr) + 1:03d}", "title": "Безопасность",
                        "description": "Хранение и обработка персональных данных по 152-ФЗ"})
            risks.append({"risk": "Требования ИБ усложнят архитектуру", "impact": "high",
                          "mitigation": "Согласовать модель угроз на этапе проектирования"})
        if any(k in text for k in ("интеграц", "api", "sap", "1с")):
            risks.append({"risk": "Внешние системы недоступны или без документации", "impact": "high",
                          "mitigation": "Запросить доступы и тестовые стенды до старта разработки"})
        total = sum(e["hours"] for t in tasks for e in t["estimates"])
        months = max(2, min(12, round(total / 400)))
        return {
            "analysis": {
                "functional_requirements": fr,
                "nonfunctional_requirements": nfr,
                "tasks": tasks,
                "architecture_options": [
                    {"name": "Модульный монолит", "recommended": True,
                     "description": "Один сервис с чёткими модулями, одна БД.",
                     "tradeoffs": ["Просто разворачивать", "Сложнее масштабировать по частям"]},
                    {"name": "Микросервисы", "recommended": False,
                     "description": "Отдельные сервисы по доменам.",
                     "tradeoffs": ["Гибкое масштабирование", "Выше стоимость сопровождения"]},
                ],
                "sizing": {"summary": "Базовый стенд для небольшой нагрузки",
                           "components": [{"name": "app", "cpu": 2, "ram_gb": 4, "storage_gb": 50},
                                          {"name": "db", "cpu": 2, "ram_gb": 4, "storage_gb": 100}]},
                "team_options": [{"name": "Базовая команда", "duration_months": months,
                                  "without_idle_time": True,
                                  "roles": ["Analyst middle 0.5", "Backend Developer middle 1.0",
                                            "Frontend Developer middle 1.0", "QA Engineer middle 0.5"]}],
                "risks": risks,
            },
            "extracted_rates": [],
            "support_scheme": "business_hours",
            "project_months": months,
        }


def build_llm(settings: Settings) -> LLM:
    if settings.llm_provider == "openai":
        return OpenAICompatibleLLM(settings)
    return MockLLM()


def parse_json_object(raw: str) -> dict:
    """Достаёт JSON-объект из ответа модели (допускает ```json-обёртку)."""
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise LLMError("Ответ модели не содержит JSON")
        try:
            value = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError("Ответ модели — невалидный JSON") from exc
    if not isinstance(value, dict):
        raise LLMError("Ожидался JSON-объект")
    return value
