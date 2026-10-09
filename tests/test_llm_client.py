"""Проверка клиента реальной модели без сети: локальный фейковый OpenAI-совместимый сервер."""
import json
import logging
import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from app import estimator
from app.config import Settings, load_dotenv
from app.llm import LLMAccessError, LLMError, OpenAICompatibleLLM
from app.service import PresaleService
from app.storage import Storage

logging.disable(logging.CRITICAL)  # тесты проверяют ошибки намеренно: не засоряем вывод логами


class QuietServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):  # клиент мог закрыть соединение по таймауту
        pass


ESTIMATE_JSON = {
    "analysis": {
        "functional_requirements": [{"code": "FR-001", "title": "Учёт заявок", "priority": "must"}],
        "nonfunctional_requirements": [],
        "tasks": [
            {"name": "Бэкенд", "description": "API", "estimates": [
                {"role": "backend", "hours": "100"}, {"role": "Тестировщик", "hours": 20},
                {"role": "Wizard", "hours": 5}]},
            {"name": "Интерфейс", "estimates": [{"role": "Frontend Developer", "hours": 40}]},
        ],
        "architecture_options": [{"name": "Монолит", "recommended": True, "description": "x"}],
        "risks": [{"risk": "Сроки", "impact": "HIGH", "mitigation": "буфер"}],
    },
    "extracted_rates": [{"role": "бэкенд", "hourly_rate": 1000}],
    "support_scheme": "24x7",
    "project_months": 4,
}


class FakeServer:
    """Запоминает запросы и отвечает по сценарию. mode: ok | reject_optional | status401 | empty | garbage_once"""

    def __init__(self, mode="ok", chat_text="Ответ модели по проекту"):
        self.mode, self.chat_text, self.requests = mode, chat_text, []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # тишина в выводе тестов
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8"))
                outer.requests.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
                if outer.mode == "slow":
                    time.sleep(3)
                if outer.mode == "status401":
                    return self._send(401, {"error": "bad key"})
                if outer.mode == "reject_optional" and ("response_format" in body or "temperature" in body):
                    return self._send(400, {"error": "unsupported parameter"})
                system = body["messages"][0]["content"]
                if "CHAT" in system:
                    content = outer.chat_text
                elif "QUESTIONS" in system:
                    content = json.dumps({"questions": ["Вопрос 1?", "Вопрос 2?"]}, ensure_ascii=False)
                elif outer.mode == "empty":
                    content = ""
                elif outer.mode == "garbage_once" and len(outer.requests) == 1:
                    content = "Извините, вот ваш ответ без JSON"
                else:
                    content = json.dumps(ESTIMATE_JSON, ensure_ascii=False)
                if body.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for word in content.split(" "):
                        chunk = {"choices": [{"delta": {"content": word + " "}}]}
                        self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
                    self.wfile.write(b"data: {\"choices\": [{\"delta\": {}}]}\n\ndata: [DONE]\n\n")
                    return
                self._send(200, {"choices": [{"message": {"content": content}}]})

            def _send(self, code, payload):
                raw = json.dumps(payload, ensure_ascii=False).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.httpd = QuietServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def make_llm(server, **overrides):
    params = {"llm_provider": "openai", "llm_api_key": "secret-key", "llm_base_url": server.url,
              "llm_model": "test-model", "llm_timeout": 5, "db_path": ":memory:"}
    params.update(overrides)
    settings = Settings(**params)
    return OpenAICompatibleLLM(settings), settings


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.servers = []

    def tearDown(self):
        for s in self.servers:
            s.close()

    def server(self, **kw):
        s = FakeServer(**kw)
        self.servers.append(s)
        return s

    def test_complete_sends_auth_model_and_json_mode(self):
        srv = self.server()
        llm, _ = make_llm(srv)
        self.assertEqual(llm.complete("sys CHAT", "привет"), "Ответ модели по проекту")
        req = srv.requests[0]
        self.assertEqual(req["path"], "/v1/chat/completions")
        self.assertEqual(req["auth"], "Bearer secret-key")
        self.assertEqual(req["body"]["model"], "test-model")
        self.assertEqual(req["body"]["temperature"], 0.2)
        self.assertNotIn("response_format", req["body"])
        llm.complete("sys", "x", json_mode=True)
        self.assertEqual(srv.requests[1]["body"]["response_format"], {"type": "json_object"})

    def test_temperature_can_be_omitted(self):
        srv = self.server()
        llm, _ = make_llm(srv, llm_temperature=None)
        llm.complete("sys CHAT", "x")
        self.assertNotIn("temperature", srv.requests[0]["body"])

    def test_retries_without_optional_params_on_400(self):
        srv = self.server(mode="reject_optional")
        llm, _ = make_llm(srv)
        text = llm.complete("sys", "x", json_mode=True)
        self.assertTrue(text)
        self.assertEqual(len(srv.requests), 2)
        self.assertNotIn("response_format", srv.requests[1]["body"])
        self.assertNotIn("temperature", srv.requests[1]["body"])

    def test_http_error_has_readable_message(self):
        srv = self.server(mode="status401")
        llm, _ = make_llm(srv)
        with self.assertRaises(LLMError) as ctx:
            llm.complete("sys", "x")
        self.assertIn("401", str(ctx.exception))
        self.assertIn("неверный ключ", str(ctx.exception))
        self.assertNotIn("secret-key", str(ctx.exception))

    def test_empty_answer_is_error(self):
        srv = self.server(mode="empty")
        llm, _ = make_llm(srv)
        with self.assertRaises(LLMError):
            llm.complete("sys", "x", json_mode=True)

    def test_unreachable_server_is_error(self):
        srv = self.server()
        llm, _ = make_llm(srv)
        srv.close()
        with self.assertRaises(LLMError) as ctx:
            llm.complete("sys", "x")
        self.assertIn("недоступен", str(ctx.exception))

    def test_total_deadline_and_not_retried(self):
        srv = self.server(mode="slow")
        llm, settings = make_llm(srv, llm_timeout=1)
        started = time.time()
        with self.assertRaises(LLMAccessError) as ctx:
            llm.complete("sys", "x", json_mode=True)
        self.assertLess(time.time() - started, 2.5)
        self.assertIn("LLM_REASONING_EFFORT", str(ctx.exception))
        svc = PresaleService(Storage(":memory:"), llm, settings)
        pid = svc.create("Система заявок", None, "Описание", [])["id"]
        before = len(srv.requests)
        with self.assertRaises(estimator.EstimationError):
            svc.run_estimate(pid)
        self.assertEqual(len(srv.requests) - before, 1)  # повторной попытки нет

    def test_reasoning_effort_sent_and_dropped_on_400(self):
        srv = self.server(mode="reject_optional")
        llm, _ = make_llm(srv, llm_reasoning_effort="none")
        llm.complete("sys", "x")
        self.assertEqual(srv.requests[0]["body"]["reasoning"], {"effort": "none"})
        self.assertNotIn("reasoning", srv.requests[1]["body"])

    def test_provider_sort_and_max_tokens_sent_and_dropped_on_400(self):
        srv = self.server(mode="reject_optional")
        llm, _ = make_llm(srv, llm_provider_sort="throughput", llm_max_tokens=500)
        llm.complete("sys", "x")
        first = srv.requests[0]["body"]
        self.assertEqual(first["provider"], {"sort": "throughput"})
        self.assertEqual(first["max_tokens"], 500)
        self.assertNotIn("provider", srv.requests[1]["body"])

    def test_timeout_message_lists_missing_hints(self):
        srv = self.server(mode="slow")
        llm, _ = make_llm(srv, llm_timeout=1)
        with self.assertRaises(LLMAccessError) as ctx:
            llm.complete("sys", "x")
        msg = str(ctx.exception)
        self.assertIn("LLM_REASONING_EFFORT=none", msg)
        self.assertIn("LLM_PROVIDER_SORT=throughput", msg)

    def test_reasoning_not_sent_by_default(self):
        srv = self.server()
        llm, _ = make_llm(srv)
        llm.complete("sys CHAT", "x")
        self.assertNotIn("reasoning", srv.requests[0]["body"])

    def test_unreachable_host_fails_fast_with_hint(self):
        srv = self.server()
        llm, _ = make_llm(srv, llm_connect_timeout=2)
        srv.close()
        started = time.time()
        with self.assertRaises(LLMAccessError) as ctx:
            llm.complete("sys", "x")
        self.assertLess(time.time() - started, 5)
        self.assertIn("нет соединения", str(ctx.exception))

    def test_requires_key(self):
        with self.assertRaises(LLMError):
            OpenAICompatibleLLM(Settings(llm_provider="openai", llm_api_key=""))

    def test_stream_yields_chunks(self):
        srv = self.server(chat_text="Первый второй третий")
        llm, _ = make_llm(srv)
        chunks = list(llm.stream("sys CHAT", "x"))
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks).strip(), "Первый второй третий")
        self.assertTrue(srv.requests[0]["body"]["stream"])


class EndToEndWithRealClientTests(unittest.TestCase):
    def setUp(self):
        self.srv = FakeServer()
        self.llm, self.settings = make_llm(self.srv)
        self.svc = PresaleService(Storage(":memory:"), self.llm, self.settings)

    def tearDown(self):
        self.srv.close()

    def test_estimate_normalizes_real_style_answer(self):
        pid = self.svc.create("Система заявок", "Клиент", "Описание проекта", [])["id"]
        result = self.svc.run_estimate(pid)
        tasks = result["analysis"]["tasks"]
        roles = {e["role"] for t in tasks for e in t["estimates"]}
        self.assertEqual(roles, {"Backend Developer", "QA Engineer", "Frontend Developer"})  # Wizard отброшен
        self.assertEqual(result["effort_budget"]["total_hours"], 160)  # 100 + 20 + 40
        self.assertEqual(result["rates"]["Backend Developer"], 1000)  # ставка из ответа модели
        self.assertEqual(result["project_months"], 4)
        self.assertEqual(result["support_budget"]["scheme"], "24x7")
        self.assertEqual(result["analysis"]["risks"][0]["impact"], "high")
        self.assertEqual(result["llm"], "openai")

    def test_estimate_retries_after_garbage(self):
        self.srv.mode = "garbage_once"
        pid = self.svc.create("Система заявок", None, "Описание", [])["id"]
        result = self.svc.run_estimate(pid)
        self.assertGreater(result["effort_budget"]["total_cost"], 0)
        self.assertEqual(len(self.srv.requests), 2)
        self.assertIn("ТОЛЬКО", self.srv.requests[1]["body"]["messages"][1]["content"])

    def test_estimate_access_error_is_not_retried(self):
        self.srv.mode = "status401"
        pid = self.svc.create("Система заявок", None, "Описание", [])["id"]
        with self.assertRaises(estimator.EstimationError):
            self.svc.run_estimate(pid)
        self.assertEqual(len(self.srv.requests), 1)

    def test_questions_come_from_model(self):
        pid = self.svc.create("Система заявок", None, "Описание", [])["id"]
        self.assertEqual(self.svc.questions(pid), ["Вопрос 1?", "Вопрос 2?"])

    def test_chat_context_contains_documents_answers_and_report(self):
        pid = self.svc.create("Система заявок", "Клиент", "Описание", [])["id"]
        self.svc.upload_document(pid, "tz.txt", "Нужна интеграция с SAP".encode())
        self.svc.save_answers(pid, [("Нагрузка?", "100 пользователей")])
        self.svc.run_estimate(pid)
        self.srv.requests.clear()
        self.assertEqual(self.svc.chat(pid, "Какие риски?"), "Ответ модели по проекту")
        prompt = self.srv.requests[0]["body"]["messages"][1]["content"]
        for part in ("Нужна интеграция с SAP", "100 пользователей", "# Пресейл", "Какие риски?"):
            self.assertIn(part, prompt)

    def test_chat_stream_saves_history_after_completion(self):
        pid = self.svc.create("Система заявок", None, "Описание", [])["id"]
        chunks = list(self.svc.chat_stream(pid, "Привет"))
        self.assertEqual("".join(chunks).strip(), "Ответ модели по проекту")
        history = self.svc.storage.list_messages(pid)
        self.assertEqual([m["role"] for m in history], ["user", "assistant"])

    def test_chat_stream_error_is_reported_and_not_saved(self):
        self.srv.mode = "status401"
        pid = self.svc.create("Система заявок", None, "Описание", [])["id"]
        text = "".join(self.svc.chat_stream(pid, "Привет"))
        self.assertIn("Ошибка модели", text)
        self.assertEqual(self.svc.storage.list_messages(pid), [])


class HelpersTests(unittest.TestCase):
    def test_canonical_role(self):
        self.assertEqual(estimator.canonical_role("backend developer"), "Backend Developer")
        self.assertEqual(estimator.canonical_role("Тестировщик"), "QA Engineer")
        self.assertEqual(estimator.canonical_role("PM"), "Project Manager")
        self.assertIsNone(estimator.canonical_role("Волшебник"))
        self.assertIsNone(estimator.canonical_role(None))

    def test_payload_without_analysis_wrapper(self):
        payload = {"tasks": [{"name": "t", "estimates": [{"role": "qa", "hours": 8}]}]}
        self.assertEqual(len(estimator.normalize_analysis(payload)["tasks"]), 1)

    def test_load_dotenv_does_not_override_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, ".env")
            with open(path, "w", encoding="utf-8-sig") as f:
                f.write('# комментарий\nTEST_PL_A="from_file"\nTEST_PL_B=from_file\n\nбез_знака\n')
            os.environ.pop("TEST_PL_A", None)
            os.environ["TEST_PL_B"] = "from_env"
            try:
                load_dotenv(path)
                self.assertEqual(os.environ["TEST_PL_A"], "from_file")
                self.assertEqual(os.environ["TEST_PL_B"], "from_env")
            finally:
                os.environ.pop("TEST_PL_A", None)
                os.environ.pop("TEST_PL_B", None)

    def test_load_dotenv_missing_file_is_ok(self):
        load_dotenv("/nonexistent/.env")


if __name__ == "__main__":
    unittest.main()
