"""Веб-интерфейс Presale Lite (Streamlit). Общается с backend по HTTP."""
import codecs
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_TOKEN = os.getenv("API_TOKEN", "demo-token")
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}
ROLES = ["Analyst", "Architect", "Backend Developer", "Frontend Developer", "QA Engineer",
         "DevOps Engineer", "Project Manager", "UX/UI Designer", "Support Engineer"]
OUTPUTS = {"requirements": "Требования", "architecture": "Архитектура", "effort": "Трудозатраты и бюджет",
           "risks": "Риски", "sizing": "Сайзинг", "support": "Поддержка и гарантия"}

st.set_page_config(page_title="Presale Lite", page_icon="📊", layout="wide")


def call(method: str, path: str, **kwargs):
    """Единая обработка запросов и ошибок API."""
    try:
        resp = requests.request(method, f"{API_URL}{path}", headers=HEADERS, timeout=420, **kwargs)
    except requests.RequestException as exc:
        st.error(f"Backend недоступен: {exc}")
        return None
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        st.error(f"Ошибка {resp.status_code}: {detail}")
        return None
    if resp.status_code == 204 or not resp.content:
        return True
    ctype = resp.headers.get("content-type", "")
    return resp.json() if "json" in ctype else resp.text


def stream_chat(pid: str, message: str):
    """Потоковый ответ чата: отдаёт текст по мере генерации моделью."""
    try:
        with requests.post(f"{API_URL}/presales/{pid}/chat/stream", headers=HEADERS,
                           json={"message": message}, stream=True, timeout=420) as resp:
            if resp.status_code >= 400:
                try:
                    detail = resp.json().get("detail", resp.text)
                except ValueError:
                    detail = resp.text
                yield f"[Ошибка {resp.status_code}: {detail}]"
                return
            decoder = codecs.getincrementaldecoder("utf-8")()
            for chunk in resp.iter_content(chunk_size=None):
                text = decoder.decode(chunk)
                if text:
                    yield text
    except requests.RequestException as exc:
        yield f"[Backend недоступен: {exc}]"


def money(v: float) -> str:
    return f"{v:,.0f} ₽".replace(",", " ")


# ---------------- sidebar: список пресейлов ----------------
st.sidebar.title("📊 Presale Lite")
health = call("GET", "/health")
if health:
    model = health.get("model")
    st.sidebar.caption(f"LLM-режим: **{health['llm']}**" + (f" · {model}" if health["llm"] != "mock" and model else ""))

presales = call("GET", "/presales") or []
labels = {p["id"]: f"{p['title']} · {p['status']}" for p in presales}
selected = st.sidebar.radio("Мои пресейлы", options=list(labels), format_func=labels.get) if labels else None

with st.sidebar.expander("➕ Новый пресейл", expanded=not presales):
    title = st.text_input("Название")
    customer = st.text_input("Заказчик")
    description = st.text_area("Описание / требования", height=140)
    outputs = st.multiselect("Нужные результаты", list(OUTPUTS), format_func=OUTPUTS.get)
    if st.button("Создать", use_container_width=True):
        created = call("POST", "/presales", json={"title": title, "customer": customer or None,
                                                  "description": description, "outputs": outputs})
        if created:
            st.rerun()

if not selected:
    st.title("Ассистент для пресейла IT-проектов")
    st.info("Создайте пресейл в боковой панели: опишите проект, загрузите ТЗ, получите оценку.")
    st.stop()

presale = call("GET", f"/presales/{selected}")
if not presale:
    st.stop()

st.title(presale["title"])
st.caption(f"Заказчик: {presale.get('customer') or '—'} · статус: {presale['status']}")
tab_in, tab_q, tab_res, tab_chat = st.tabs(["1. Исходные данные", "2. Вопросы", "3. Результат", "4. Чат"])

# ---------------- 1. исходные данные ----------------
with tab_in:
    left, right = st.columns(2)
    with left:
        st.subheader("Документы")
        for d in presale["documents"]:
            st.write(f"📄 {d['filename']}")
        upload = st.file_uploader("Загрузить ТЗ (txt, md, docx, pdf)", type=["txt", "md", "docx", "pdf"])
        if upload and st.button("Загрузить файл"):
            if call("POST", f"/presales/{selected}/documents", files={"file": (upload.name, upload.getvalue())}):
                st.success("Файл загружен")
                st.rerun()
    with right:
        st.subheader("Ставки специалистов (необязательно)")
        st.caption("Если пусто — используются рыночные допущения.")
        rates = st.data_editor(
            presale["rates"] or [{"role": ROLES[2], "hourly_rate": 3600.0}], num_rows="dynamic",
            column_config={"role": st.column_config.SelectboxColumn("Роль", options=ROLES),
                           "hourly_rate": st.column_config.NumberColumn("₽/час", min_value=1)},
            key="rates_editor")
        if st.button("Сохранить ставки"):
            clean = [r for r in rates if r.get("role") and r.get("hourly_rate")]
            if call("PUT", f"/presales/{selected}/rates", json=clean):
                st.success("Ставки сохранены")

# ---------------- 2. вопросы ----------------
with tab_q:
    if st.button("Сформировать вопросы преданализа"):
        data = call("POST", f"/presales/{selected}/questions")
        if data:
            st.session_state[f"q_{selected}"] = data["questions"]
    questions = st.session_state.get(f"q_{selected}", [])
    answers = [{"question": q, "answer": st.text_area(q, key=f"a_{selected}_{i}")}
               for i, q in enumerate(questions)]
    if questions and st.button("Сохранить ответы"):
        if call("PUT", f"/presales/{selected}/answers", json={"items": answers}):
            st.success("Ответы сохранены")

# ---------------- 3. результат ----------------
with tab_res:
    if st.button("🚀 Рассчитать пресейл", type="primary"):
        with st.spinner("Формирую оценку..."):
            if call("POST", f"/presales/{selected}/estimate"):
                st.rerun()
    result = presale.get("result")
    if result:
        eff = result["effort_budget"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Разработка", money(eff["total_cost"]))
        c2.metric("Трудозатраты", f"{eff['total_hours']:g} ч")
        c3.metric("Гарантия / год", money(result["warranty_budget"]["annual_cost"]))
        c4.metric("Поддержка / год", money(result["support_budget"]["annual_cost"]))
        a = result["analysis"]
        st.subheader("Декомпозиция")
        st.dataframe([{"Задача": t["name"], "Роль": e["role"], "Часы": e["hours"]}
                      for t in a["tasks"] for e in t["estimates"]], use_container_width=True)
        st.subheader("Расходы по месяцам")
        st.bar_chart({f"М{m['month']}": m["total"] for m in result["monthly_expenses"]})
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Архитектура")
            for o in a["architecture_options"]:
                st.markdown(f"**{o['name']}**{' ✅' if o.get('recommended') else ''} — {o.get('description', '')}")
        with col_b:
            st.subheader("Риски")
            for r in a["risks"]:
                st.markdown(f"- **[{r['impact']}]** {r['risk']} → {r['mitigation']}")
        with st.expander("Требования (ФТ/НФТ)"):
            for r in a["functional_requirements"]:
                st.markdown(f"**{r.get('code', '')}** {r.get('title', '')} _({r.get('priority', '')})_")
            for r in a["nonfunctional_requirements"]:
                st.markdown(f"**{r.get('code', '')}** {r.get('title', '')}: {r.get('description', '')}")
        with st.expander("Сайзинг и команда"):
            sizing = a.get("sizing") or {}
            if sizing.get("summary"):
                st.write(sizing["summary"])
            if sizing.get("components"):
                st.dataframe(sizing["components"], use_container_width=True)
            for team in a.get("team_options") or []:
                roles = ", ".join(str(x) for x in team.get("roles") or [])
                st.markdown(f"**{team.get('name', '')}** ({team.get('duration_months', '?')} мес.): {roles}")
        report_md = call("GET", f"/presales/{selected}/report")
        if report_md:
            st.download_button("⬇️ Скачать отчёт (.md)", report_md,
                               file_name="presale_report.md", mime="text/markdown")
    else:
        st.info("Оценка ещё не выполнена.")

# ---------------- 4. чат ----------------
with tab_chat:
    for m in presale["messages"]:
        st.chat_message(m["role"]).write(m["content"])
    if prompt := st.chat_input("Вопрос по этому пресейлу"):
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            st.write_stream(stream_chat(selected, prompt))
