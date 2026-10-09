# Presale Lite — AI-ассистент для пресейла IT-проектов

Упрощённая самостоятельная реализация идеи «Presales Assistant»: по описанию и документам заказчика
система формирует требования, декомпозицию задач, архитектуру, команду, бюджет (разработка, поддержка,
гарантия, расходы по месяцам) и риски, а затем отвечает на вопросы по результату в чате.

## Быстрый старт

```bash
cp .env.example .env          # по умолчанию LLM_PROVIDER=mock — работает без ключа
docker compose up --build
```
* Интерфейс: http://localhost:8501
* API и документация: http://localhost:8000/docs (токен — `API_TOKEN` из `.env`)

Без Docker:
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload                 # терминал 1
streamlit run ui/streamlit_app.py             # терминал 2
```

## Подключение реальной модели
По умолчанию работает офлайн-режим `mock`. Чтобы использовать настоящую модель, создайте в папке проекта
файл `.env` (он в `.gitignore`, ключ не попадёт в репозиторий) и перезапустите backend:

```ini
LLM_PROVIDER=openai
LLM_API_KEY=ваш_ключ
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=openai/gpt-4o-mini
```

Подойдёт любой OpenAI-совместимый API. Примеры `LLM_BASE_URL`:
* OpenRouter — `https://openrouter.ai/api/v1` (модель в формате `провайдер/модель`);
* Gemini (Google AI Studio) — `https://generativelanguage.googleapis.com/v1beta/openai`;
* OpenAI — `https://api.openai.com/v1`.

Если модель не принимает `temperature`, задайте пустое `LLM_TEMPERATURE=`. Проверка: `http://localhost:8000/health`
покажет `"llm": "openai"` и имя модели. Чат отвечает потоком (текст появляется по мере генерации).
Ключ не показывайте на скриншотах и не публикуйте.

## Тесты
```bash
pip install -r requirements-dev.txt && pytest -q
# без установки зависимостей ядро проверяется так:
python -m unittest discover -s tests
```

## Структура
```
app/pricing.py     расчёт бюджета (чистые функции)
app/llm.py         провайдеры LLM: mock и OpenAI-совместимый (с потоковым выводом)
app/estimator.py   промпты, проверка ответа модели, сборка оценки
app/documents.py   извлечение текста: txt, md, docx, pdf + валидация
app/storage.py     SQLite
app/service.py     сценарии (без привязки к веб-фреймворку)
app/api.py         FastAPI (токен, обработка ошибок)
ui/streamlit_app.py веб-интерфейс
```
Демо-документ: `docs/sample_tz.txt`.
