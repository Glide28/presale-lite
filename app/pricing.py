"""Расчёт бюджета: трудозатраты, поддержка, гарантия, расходы по месяцам.

Чистые функции без внешних зависимостей — легко тестируются.
"""
from __future__ import annotations

DEFAULT_RATES: dict[str, float] = {
    "Analyst": 2800,
    "Architect": 5200,
    "Backend Developer": 3600,
    "Frontend Developer": 3400,
    "QA Engineer": 2600,
    "DevOps Engineer": 3800,
    "Project Manager": 3200,
    "UX/UI Designer": 3000,
    "Support Engineer": 2400,
}
FALLBACK_RATE = 3000
WARRANTY_PERCENT = 8
SUPPORT_SCHEMES = {
    "business_hours": ("3 линия, рабочие дни 06:00-18:00", 8 * 5 * 4),
    "24x7": ("1/2/3 линии, 24x7x365", round(24 * 365 / 12 * 3, 2)),
}


def normalize_rates(custom: list[dict] | None) -> dict[str, float]:
    """Рыночные ставки по умолчанию + ставки пользователя поверх них."""
    rates = dict(DEFAULT_RATES)
    for item in custom or []:
        role = str(item.get("role", "")).strip()
        rate = float(item.get("hourly_rate", 0))
        if role and rate > 0:
            rates[role] = rate
    return rates


def effort_budget(tasks: list[dict], rates: dict[str, float]) -> dict:
    total_hours = 0.0
    total_cost = 0.0
    by_role: dict[str, dict] = {}
    for task in tasks:
        for est in task.get("estimates", []):
            hours = float(est.get("hours", 0))
            if hours <= 0:
                continue
            role = est["role"]
            cost = hours * rates.get(role, FALLBACK_RATE)
            total_hours += hours
            total_cost += cost
            bucket = by_role.setdefault(role, {"hours": 0.0, "cost": 0.0})
            bucket["hours"] += hours
            bucket["cost"] = round(bucket["cost"] + cost, 2)
    return {"total_hours": total_hours, "total_cost": round(total_cost, 2), "by_role": by_role}


def support_budget(scheme: str, rates: dict[str, float]) -> dict:
    key = scheme if scheme in SUPPORT_SCHEMES else "business_hours"
    title, monthly_hours = SUPPORT_SCHEMES[key]
    monthly_cost = monthly_hours * rates.get("Support Engineer", DEFAULT_RATES["Support Engineer"])
    return {
        "scheme": key,
        "title": title,
        "monthly_hours": round(monthly_hours, 2),
        "monthly_cost": round(monthly_cost, 2),
        "annual_cost": round(monthly_cost * 12, 2),
    }


def warranty_budget(development_cost: float, percent: float = WARRANTY_PERCENT) -> dict:
    return {"percent": percent, "annual_cost": round(development_cost * percent / 100, 2)}


def monthly_expenses(development_cost: float, warranty_annual: float, months: int) -> list[dict]:
    months = max(1, int(months))
    dev_month = development_cost / months
    warranty_month = warranty_annual / 12
    return [
        {
            "month": m,
            "development": round(dev_month, 2),
            "warranty_reserve": round(warranty_month, 2),
            "total": round(dev_month + warranty_month, 2),
        }
        for m in range(1, months + 1)
    ]
