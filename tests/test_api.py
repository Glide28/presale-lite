"""Тесты HTTP-слоя. Пропускаются, если FastAPI не установлен."""
import unittest

try:
    from fastapi.testclient import TestClient
except (ImportError, RuntimeError):  # нет fastapi или httpx для TestClient
    TestClient = None

from app.config import Settings
from app.llm import MockLLM
from app.service import PresaleService
from app.storage import Storage

AUTH = {"Authorization": "Bearer test-token"}


@unittest.skipIf(TestClient is None, "нужны fastapi и httpx (pip install httpx)")
class ApiTests(unittest.TestCase):
    def setUp(self):
        from app.api import create_app
        settings = Settings(db_path=":memory:", api_token="test-token")
        svc = PresaleService(Storage(":memory:"), MockLLM(), settings)
        self.client = TestClient(create_app(settings, svc))

    def test_health_is_public(self):
        self.assertEqual(self.client.get("/health").json()["status"], "ok")

    def test_requires_token(self):
        self.assertEqual(self.client.get("/presales").status_code, 401)
        self.assertEqual(self.client.get("/presales", headers={"Authorization": "Bearer x"}).status_code, 401)

    def test_full_flow(self):
        r = self.client.post("/presales", headers=AUTH,
                             json={"title": "Портал", "description": "Нужна интеграция API", "outputs": ["effort"]})
        self.assertEqual(r.status_code, 201)
        pid = r.json()["id"]
        up = self.client.post(f"/presales/{pid}/documents", headers=AUTH,
                              files={"file": ("tz.txt", "Требования".encode(), "text/plain")})
        self.assertEqual(up.status_code, 201)
        est = self.client.post(f"/presales/{pid}/estimate", headers=AUTH)
        self.assertEqual(est.status_code, 200)
        self.assertGreater(est.json()["effort_budget"]["total_cost"], 0)
        self.assertIn("# Пресейл", self.client.get(f"/presales/{pid}/report", headers=AUTH).text)
        chat = self.client.post(f"/presales/{pid}/chat", headers=AUTH, json={"message": "Риски?"})
        self.assertEqual(chat.status_code, 200)

    def test_chat_stream(self):
        pid = self.client.post("/presales", headers=AUTH,
                               json={"title": "Потоковый чат", "description": "Интеграция API"}).json()["id"]
        self.client.post(f"/presales/{pid}/estimate", headers=AUTH)
        r = self.client.post(f"/presales/{pid}/chat/stream", headers=AUTH, json={"message": "Какие риски?"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("Риски", r.text)
        detail = self.client.get(f"/presales/{pid}", headers=AUTH).json()
        self.assertEqual([m["role"] for m in detail["messages"]], ["user", "assistant"])
        for bad in ("", "   "):
            r = self.client.post(f"/presales/{pid}/chat/stream", headers=AUTH, json={"message": bad})
            self.assertEqual(r.status_code, 422)

    def test_health_reports_model(self):
        self.assertEqual(self.client.get("/health").json()["model"], "mock")

    def test_errors(self):
        self.assertEqual(self.client.get("/presales/nope", headers=AUTH).status_code, 404)
        self.assertEqual(self.client.post("/presales", headers=AUTH, json={"title": "ab"}).status_code, 422)
        pid = self.client.post("/presales", headers=AUTH, json={"title": "Проект"}).json()["id"]
        self.assertEqual(self.client.post(f"/presales/{pid}/estimate", headers=AUTH).status_code, 422)
        bad = self.client.post(f"/presales/{pid}/documents", headers=AUTH,
                               files={"file": ("a.exe", b"x", "application/octet-stream")})
        self.assertEqual(bad.status_code, 422)
        self.assertEqual(self.client.get(f"/presales/{pid}/report", headers=AUTH).status_code, 409)
