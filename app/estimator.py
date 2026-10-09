"""Оркестрация пресейл-оценки: LLM -> проверка результата -> расчёт бюджета."""
from __future__ import annotations

import logging
from collections.abc import Iterator

from app import pricing
from app.llm import LLM, LLMAccessError, LLMError, parse_json_object

log = logging.getLogger("presale.estimator")

OUTPUT_SECTIONS = {
    "requirements", "architecture", "effort", "risks", "sizing", "support",
}

ESTIMATE_SYSTEM = """Ты AI-ассистент для пресейла: системный аналитик, архитектор и оценщик IT-проектов.
По описанию и документам заказчика сформируй: функциональные и нефункциональные требования,
декомпозицию задач с оценкой часов по ролям, варианты архитектуры, сайзинг, состав команды и риски.
Используй только роли: Analyst, Architect, Backend Developer, Frontend Developer, QA Engineer,
DevOps Engineer, Project Manager, UX/UI Designer, Support Engineer.
Верни ТОЛЬКО валидный JSON:
{"analysis": {"functional_requirements": [{"code": "FR-001", "title": "", "priority": "must|should|could"}],
"nonfunctional_requirements": [{"code": "NFR-001", "title": "", "description": ""}],
"tasks": [{"name": "", "description": "", "estimates": [{"role": "Backend Developer", "hours": 40}]}],
"architecture_options": [{"name": "", "recommended": true, "description": "", "tradeoffs": [""]}],
"sizing": {"summary": "", "components": [{"name": "", "cpu": 2, "ram_gb": 4, "storage_gb": 50}]},
"team_options": [{"name": "", "duration_months": 6, "without_idle_time": true, "roles": [""]}],
"risks": [{"risk": "", "impact": "low|medium|high", "mitigation": ""}]},
"extracted_rates": [{"role": "Backend Developer", "hourly_rate": 1000}],
"support_scheme": "business_hours|24x7", "project_months": 6}
Если ставок нет — не спрашивай, используй рыночные допущения. Если данных мало — делай допущения явно.
ВАЖНО: пиши компактно, чтобы ответ был коротким. Не более 8 функциональных и 5 нефункциональных требований,
не более 8 задач (описание задачи — одна короткая фраза, не более 3 ролей в задаче), 2 варианта архитектуры
(описание до 15 слов, 2 компромисса), 1 вариант команды, не более 6 рисков (одна фраза на риск и на меру).
Только JSON, без пояснений и без markdown."""

QUESTIONS_SYSTEM = """Ты AI-ассистент для пресейла. Режим QUESTIONS.
Сформируй 5-8 уточняющих вопросов заказчику (интеграции, безопасность, нагрузка, сроки, поддержка).
Не спрашивай то, что уже явно указано. Верни только JSON: {"questions": ["...", "..."]}"""

CHAT_SYSTEM = """Ты AI-ассистент для пресейла. Режим CHAT.
Отвечай по-русски, кратко и по делу, опираясь ТОЛЬКО на переданный контекст пресейла.
Если данных не хватает — прямо скажи, что нужно уточнить."""

VALID_ROLES = set(pricing.DEFAULT_RATES)
VALID_IMPACT = {"low", "medium", "high"}

_ROLE_ALIASES = {
    "аналитик": "Analyst", "системный аналитик": "Analyst", "бизнес-аналитик": "Analyst",
    "архитектор": "Architect", "backend": "Backend Developer", "бэкенд": "Backend Developer",
    "backend developer": "Backend Developer", "backend-разработчик": "Backend Developer",
    "frontend": "Frontend Developer", "фронтенд": "Frontend Developer",
    "frontend developer": "Frontend Developer", "frontend-разработчик": "Frontend Developer",
    "qa": "QA Engineer", "тестировщик": "QA Engineer", "qa engineer": "QA Engineer",
    "devops": "DevOps Engineer", "devops engineer": "DevOps Engineer",
    "pm": "Project Manager", "менеджер проекта": "Project Manager", "руководитель проекта": "Project Manager",
    "project manager": "Project Manager", "дизайнер": "UX/UI Designer", "ux/ui designer": "UX/UI Designer",
    "ui/ux designer": "UX/UI Designer", "support": "Support Engineer", "support engineer": "Support Engineer",
}


def canonical_role(name: object) -> str | None:
    """Приводит название роли от модели к одной из допустимых (регистр, синонимы)."""
    text = str(name or "").strip()
    if text in VALID_ROLES:
        return text
    lowered = text.lower()
    for role in VALID_ROLES:
        if role.lower() == lowered:
            return role
    return _ROLE_ALIASES.get(lowered)


class EstimationError(RuntimeError):
    """Не удалось получить корректную оценку."""


def default_questions() -> list[str]:
    return [
        "Какая бизнес-цель проекта и KPI?",
        "Кто пользователи системы и какие ключевые сценарии?",
        "Какие внешние системы и интеграции нужны?",
        "Есть ли ограничения по размещению (on-premise, закрытый контур)?",
        "Какие требования к безопасности и персональным данным?",
        "Какая ожидаемая нагрузка?",
        "Какие сроки и бюджетные ограничения?",
        "Какая схема поддержки нужна после запуска и нужна ли гарантия?",
    ]


def generate_questions(llm: LLM, description: str) -> list[str]:
    try:
        raw = llm.complete(QUESTIONS_SYSTEM, f"Описание проекта:\n{description or 'нет данных'}",
                           json_mode=True)
        questions = parse_json_object(raw).get("questions", [])
    except LLMError as exc:
        log.warning("questions fallback: %s", exc)
        return default_questions()
    cleaned = [str(q).strip() for q in questions if str(q).strip()]
    return cleaned or default_questions()


def build_user_prompt(title: str, customer: str | None, description: str,
                      documents: list[tuple[str, str]], answers: list[tuple[str, str]],
                      outputs: list[str], rates: list[dict]) -> str:
    parts = [f"Проект: {title}", f"Заказчик: {customer or 'не указан'}",
             f"Описание:\n{description or '—'}"]
    for name, text in documents:
        parts.append(f"Документ «{name}»:\n{text}")
    if answers:
        parts.append("Ответы на вопросы преданализа:\n" +
                     "\n".join(f"- {q}: {a}" for q, a in answers if a))
    if rates:
        parts.append("Ставки пользователя: " +
                     ", ".join(f"{r['role']} {r['hourly_rate']} руб/час" for r in rates))
    if outputs:
        parts.append("Нужные результаты: " + ", ".join(outputs))
    return "\n\n".join(parts)


def normalize_analysis(payload: dict) -> dict:
    """Приводит ответ модели к безопасной структуре; бросает EstimationError, если задач нет."""
    analysis = payload.get("analysis")
    if not isinstance(analysis, dict) and isinstance(payload.get("tasks"), list):
        analysis = payload  # модель вернула разделы на верхнем уровне
    if not isinstance(analysis, dict):
        raise EstimationError("В ответе модели нет раздела analysis")

    tasks: list[dict] = []
    for task in analysis.get("tasks") or []:
        estimates = []
        for est in task.get("estimates") or []:
            role = canonical_role(est.get("role"))
            try:
                hours = float(est.get("hours", 0))
            except (TypeError, ValueError):
                continue
            if role and 0 < hours <= 5000:
                estimates.append({"role": role, "hours": hours})
        if task.get("name") and estimates:
            tasks.append({"name": str(task["name"]), "description": str(task.get("description", "")),
                          "estimates": estimates})
    if not tasks:
        raise EstimationError("Модель не вернула ни одной корректной задачи с оценкой")

    risks = []
    for risk in analysis.get("risks") or []:
        if risk.get("risk"):
            impact = str(risk.get("impact", "medium")).lower()
            risks.append({"risk": str(risk["risk"]),
                          "impact": impact if impact in VALID_IMPACT else "medium",
                          "mitigation": str(risk.get("mitigation", ""))})

    def dicts(key: str) -> list[dict]:
        value = analysis.get(key)
        return [x for x in value if isinstance(x, dict)] if isinstance(value, list) else []

    sizing = analysis.get("sizing")
    return {
        "functional_requirements": dicts("functional_requirements"),
        "nonfunctional_requirements": dicts("nonfunctional_requirements"),
        "tasks": tasks,
        "architecture_options": dicts("architecture_options"),
        "sizing": sizing if isinstance(sizing, dict) else {},
        "team_options": dicts("team_options"),
        "risks": risks,
    }


def _ask_analysis(llm: LLM, user_prompt: str) -> tuple[dict, dict]:
    raw = llm.complete(ESTIMATE_SYSTEM, user_prompt, json_mode=True)
    payload = parse_json_object(raw)
    return payload, normalize_analysis(payload)


def estimate(llm: LLM, user_prompt: str, user_rates: list[dict] | None = None) -> dict:
    """Полный цикл: запрос к модели, проверка, расчёт бюджета. Одна повторная попытка при мусорном ответе."""
    try:
        try:
            payload, analysis = _ask_analysis(llm, user_prompt)
        except LLMAccessError:
            raise  # сеть, ключ, лимиты, таймаут: повторять бессмысленно
        except (LLMError, EstimationError) as first:
            log.warning("estimate retry after: %s", first)
            payload, analysis = _ask_analysis(
                llm, user_prompt + "\n\nВАЖНО: верни ТОЛЬКО один валидный JSON-объект строго по схеме, без пояснений.")
    except LLMError as exc:
        raise EstimationError(str(exc)) from exc

    extracted = [{"role": canonical_role(r.get("role")), "hourly_rate": r.get("hourly_rate")}
                 for r in payload.get("extracted_rates") or []
                 if isinstance(r, dict) and canonical_role(r.get("role"))]
    rates = pricing.normalize_rates((user_rates or []) + extracted)

    try:
        months = int(payload.get("project_months") or 6)
    except (TypeError, ValueError):
        months = 6
    months = max(1, min(36, months))

    effort = pricing.effort_budget(analysis["tasks"], rates)
    support = pricing.support_budget(str(payload.get("support_scheme", "business_hours")), rates)
    warranty = pricing.warranty_budget(effort["total_cost"])
    monthly = pricing.monthly_expenses(effort["total_cost"], warranty["annual_cost"], months)

    return {
        "analysis": analysis,
        "rates": rates,
        "project_months": months,
        "effort_budget": effort,
        "support_budget": support,
        "warranty_budget": warranty,
        "monthly_expenses": monthly,
        "llm": llm.name,
    }


def _chat_prompt(context: str, history: list[dict], message: str) -> str:
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history[-10:])
    return f"Контекст пресейла:\n{context}\n\nИстория диалога:\n{transcript}\n\nВопрос: {message}"


def chat_answer(llm: LLM, context: str, history: list[dict], message: str) -> str:
    try:
        return llm.complete(CHAT_SYSTEM, _chat_prompt(context, history, message))
    except LLMError as exc:
        raise EstimationError(str(exc)) from exc


def chat_stream(llm: LLM, context: str, history: list[dict], message: str) -> Iterator[str]:
    try:
        yield from llm.stream(CHAT_SYSTEM, _chat_prompt(context, history, message))
    except LLMError as exc:
        raise EstimationError(str(exc)) from exc
