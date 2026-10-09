"""Формирование итогового отчёта по пресейлу в Markdown."""
from __future__ import annotations


def _money(value: float) -> str:
    return f"{value:,.0f} ₽".replace(",", " ")


def render_markdown(presale: dict, result: dict) -> str:
    a = result["analysis"]
    effort = result["effort_budget"]
    lines = [
        f"# Пресейл: {presale['title']}",
        f"Заказчик: {presale.get('customer') or 'не указан'}",
        "",
        "## Требования",
        *(f"- **{r.get('code', '')}** {r.get('title', '')} ({r.get('priority', '')})"
          for r in a["functional_requirements"]),
        *(f"- **{r.get('code', '')}** {r.get('title', '')}: {r.get('description', '')}"
          for r in a["nonfunctional_requirements"]),
        "",
        "## Декомпозиция и трудозатраты",
        "| Задача | Роль | Часы |",
        "|---|---|---|",
    ]
    for task in a["tasks"]:
        for est in task["estimates"]:
            lines.append(f"| {task['name']} | {est['role']} | {est['hours']:g} |")
    lines += ["", "## Архитектура"]
    for opt in a["architecture_options"]:
        mark = " (рекомендуется)" if opt.get("recommended") else ""
        lines.append(f"- **{opt.get('name', '')}**{mark}: {opt.get('description', '')}")
    sizing = a.get("sizing") if isinstance(a.get("sizing"), dict) else {}
    components = [c for c in sizing.get("components") or [] if isinstance(c, dict)]
    if sizing.get("summary") or components:
        lines += ["", "## Сайзинг"]
        if sizing.get("summary"):
            lines.append(str(sizing["summary"]))
        lines += [f"- {c.get('name', '')}: CPU {c.get('cpu', '?')}, RAM {c.get('ram_gb', '?')} ГБ, "
                  f"диск {c.get('storage_gb', '?')} ГБ" for c in components]
    teams = [t for t in a.get("team_options") or [] if isinstance(t, dict)]
    if teams:
        lines += ["", "## Команда"]
        for t in teams:
            roles = ", ".join(str(r) for r in t.get("roles") or [])
            lines.append(f"- **{t.get('name', '')}** ({t.get('duration_months', '?')} мес.): {roles}")
    lines += [
        "",
        "## Бюджет",
        f"- Разработка: {_money(effort['total_cost'])} ({effort['total_hours']:g} ч)",
        (f"- Гарантия ({result['warranty_budget']['percent']}%/год): "
         f"{_money(result['warranty_budget']['annual_cost'])}"),
        (f"- Поддержка ({result['support_budget']['title']}): "
         f"{_money(result['support_budget']['annual_cost'])}/год"),
        "",
        "### Расходы по месяцам",
        *(f"- Месяц {m['month']}: {_money(m['total'])}" for m in result["monthly_expenses"]),
        "",
        "## Риски",
        *(f"- [{r['impact']}] {r['risk']} — {r['mitigation']}" for r in a["risks"]),
    ]
    return "\n".join(lines)
