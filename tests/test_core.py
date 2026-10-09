"""Тесты ядра. Запуск: python -m unittest discover -s tests -v  (или pytest)."""
import io
import json
import logging
import unittest

from app import documents, estimator, pricing, report
from app.config import Settings
from app.llm import LLMError, MockLLM, parse_json_object
from app.service import NotFound, PresaleService, ValidationError
from app.storage import Storage

logging.disable(logging.CRITICAL)


def make_service() -> PresaleService:
    settings = Settings(db_path=":memory:")
    return PresaleService(Storage(":memory:"), MockLLM(), settings)


class PricingTests(unittest.TestCase):
    def test_custom_rate_overrides_default(self):
        rates = pricing.normalize_rates([{"role": "Backend Developer", "hourly_rate": 1000}])
        self.assertEqual(rates["Backend Developer"], 1000)
        self.assertEqual(rates["Architect"], pricing.DEFAULT_RATES["Architect"])

    def test_invalid_custom_rates_are_ignored(self):
        rates = pricing.normalize_rates([{"role": "", "hourly_rate": 5}, {"role": "QA Engineer", "hourly_rate": 0}])
        self.assertEqual(rates["QA Engineer"], pricing.DEFAULT_RATES["QA Engineer"])

    def test_effort_budget(self):
        tasks = [{"estimates": [{"role": "Backend Developer", "hours": 10},
                                {"role": "QA Engineer", "hours": 5}]}]
        rates = {"Backend Developer": 100, "QA Engineer": 50}
        result = pricing.effort_budget(tasks, rates)
        self.assertEqual(result["total_hours"], 15)
        self.assertEqual(result["total_cost"], 1250)
        self.assertEqual(result["by_role"]["QA Engineer"]["cost"], 250)

    def test_unknown_role_uses_fallback_rate(self):
        result = pricing.effort_budget([{"estimates": [{"role": "Wizard", "hours": 2}]}], {})
        self.assertEqual(result["total_cost"], 2 * pricing.FALLBACK_RATE)

    def test_support_budget_business_hours(self):
        s = pricing.support_budget("business_hours", {"Support Engineer": 1000})
        self.assertEqual(s["monthly_hours"], 160)
        self.assertEqual(s["annual_cost"], 160 * 1000 * 12)

    def test_support_budget_unknown_scheme_falls_back(self):
        self.assertEqual(pricing.support_budget("???", {})["scheme"], "business_hours")

    def test_warranty_and_monthly(self):
        w = pricing.warranty_budget(1_000_000)
        self.assertEqual(w["annual_cost"], 80_000)
        months = pricing.monthly_expenses(1_200_000, w["annual_cost"] * 1.5, 6)
        self.assertEqual(len(months), 6)
        self.assertAlmostEqual(months[0]["development"], 200_000)

    def test_monthly_expenses_zero_months_is_safe(self):
        self.assertEqual(len(pricing.monthly_expenses(100, 12, 0)), 1)


class DocumentTests(unittest.TestCase):
    def test_txt_utf8(self):
        self.assertEqual(documents.extract_text("a.txt", "Привет".encode("utf-8")), "Привет")

    def test_txt_cp1251(self):
        self.assertEqual(documents.extract_text("a.txt", "Привет".encode("cp1251")), "Привет")

    def test_rejects_extension(self):
        with self.assertRaises(documents.DocumentError):
            documents.extract_text("virus.exe", b"x")

    def test_rejects_empty_and_large(self):
        with self.assertRaises(documents.DocumentError):
            documents.extract_text("a.txt", b"")
        with self.assertRaises(documents.DocumentError):
            documents.extract_text("a.txt", b"x" * 20, max_bytes=10)

    def test_docx_roundtrip(self):
        from docx import Document
        doc = Document()
        doc.add_paragraph("Требования заказчика")
        buf = io.BytesIO()
        doc.save(buf)
        self.assertIn("Требования заказчика", documents.extract_text("t.docx", buf.getvalue()))

    def test_broken_docx(self):
        with self.assertRaises(documents.DocumentError):
            documents.extract_text("t.docx", b"not a docx")


class JsonParsingTests(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(parse_json_object('{"a": 1}'), {"a": 1})

    def test_fenced(self):
        self.assertEqual(parse_json_object('```json\n{"a": 1}\n```'), {"a": 1})

    def test_with_noise(self):
        self.assertEqual(parse_json_object('Вот результат: {"a": 1} Готово'), {"a": 1})

    def test_invalid(self):
        for bad in ("совсем не json", "[1, 2]", '{"a": '):
            with self.assertRaises(LLMError):
                parse_json_object(bad)


class EstimatorTests(unittest.TestCase):
    def test_mock_estimate_has_budget(self):
        prompt = estimator.build_user_prompt("Проект", None, "Нужна интеграция с 1С и безопасность", [], [], [], [])
        result = estimator.estimate(MockLLM(), prompt)
        self.assertGreater(result["effort_budget"]["total_cost"], 0)
        self.assertEqual(len(result["monthly_expenses"]), result["project_months"])
        names = [t["name"] for t in result["analysis"]["tasks"]]
        self.assertIn("Интеграции с внешними системами", names)

    def test_user_rates_applied(self):
        prompt = estimator.build_user_prompt("P", None, "описание", [], [], [], [])
        cheap = estimator.estimate(MockLLM(), prompt, [{"role": "Backend Developer", "hourly_rate": 1}])
        normal = estimator.estimate(MockLLM(), prompt)
        self.assertLess(cheap["effort_budget"]["total_cost"], normal["effort_budget"]["total_cost"])

    def test_normalize_drops_invalid_estimates(self):
        payload = {"analysis": {"tasks": [
            {"name": "ok", "estimates": [{"role": "QA Engineer", "hours": 10},
                                         {"role": "Hacker", "hours": 10},
                                         {"role": "QA Engineer", "hours": -5},
                                         {"role": "QA Engineer", "hours": "abc"}]},
            {"name": "empty", "estimates": []}]}}
        analysis = estimator.normalize_analysis(payload)
        self.assertEqual(len(analysis["tasks"]), 1)
        self.assertEqual(analysis["tasks"][0]["estimates"], [{"role": "QA Engineer", "hours": 10.0}])

    def test_normalize_requires_tasks(self):
        with self.assertRaises(estimator.EstimationError):
            estimator.normalize_analysis({"analysis": {"tasks": []}})
        with self.assertRaises(estimator.EstimationError):
            estimator.normalize_analysis({})

    def test_invalid_impact_defaults_to_medium(self):
        payload = {"analysis": {"tasks": [{"name": "t", "estimates": [{"role": "QA Engineer", "hours": 1}]}],
                                "risks": [{"risk": "r", "impact": "catastrophic"}]}}
        self.assertEqual(estimator.normalize_analysis(payload)["risks"][0]["impact"], "medium")

    def test_broken_llm_answer_raises(self):
        class Broken:
            name = "broken"
            def complete(self, system, user, *, json_mode=False):
                return "это не JSON"
        with self.assertRaises(estimator.EstimationError):
            estimator.estimate(Broken(), "x")

    def test_questions_fallback_on_llm_error(self):
        class Down:
            name = "down"
            def complete(self, system, user, *, json_mode=False):
                raise LLMError("down")
        self.assertEqual(estimator.generate_questions(Down(), "x"), estimator.default_questions())


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.svc = make_service()

    def test_full_flow(self):
        p = self.svc.create("Портал заказчика", "ООО Ромашка", "Нужен портал с отчётами и интеграцией API", ["effort"])
        pid = p["id"]
        self.svc.upload_document(pid, "tz.txt", "Требования: безопасность и персональные данные".encode())
        self.svc.set_rates(pid, [{"role": "Backend Developer", "hourly_rate": 1500}])
        questions = self.svc.questions(pid)
        self.assertTrue(questions)
        self.svc.save_answers(pid, [(questions[0], "Интеграция с 1С")])
        result = self.svc.run_estimate(pid)
        self.assertIn("# Пресейл: Портал заказчика", result["report_markdown"])
        self.assertEqual(self.svc.storage.get_presale(pid)["status"], "estimated")
        self.assertTrue(self.svc.chat(pid, "Какие риски?"))
        self.assertEqual(len(self.svc.storage.list_messages(pid)), 2)

    def test_validation(self):
        with self.assertRaises(ValidationError):
            self.svc.create("ab", None, "", [])
        with self.assertRaises(ValidationError):
            self.svc.create("Нормальное имя", None, "", ["nonsense"])

    def test_estimate_requires_input(self):
        pid = self.svc.create("Пустой проект", None, "", [])["id"]
        with self.assertRaises(ValidationError):
            self.svc.run_estimate(pid)

    def test_bad_upload_and_rates(self):
        pid = self.svc.create("Проект X", None, "описание", [])["id"]
        with self.assertRaises(ValidationError):
            self.svc.upload_document(pid, "a.exe", b"x")
        with self.assertRaises(ValidationError):
            self.svc.set_rates(pid, [{"role": "QA Engineer", "hourly_rate": -1}])

    def test_not_found_and_delete(self):
        with self.assertRaises(NotFound):
            self.svc.storage.get_presale("nope")
        pid = self.svc.create("Удаляемый", None, "d", [])["id"]
        self.svc.storage.delete_presale(pid)
        with self.assertRaises(NotFound):
            self.svc.storage.get_presale(pid)

    def test_chat_rejects_empty(self):
        pid = self.svc.create("Проект Y", None, "d", [])["id"]
        with self.assertRaises(ValidationError):
            self.svc.chat(pid, "   ")


class ReportTests(unittest.TestCase):
    def test_report_tolerates_junk_from_model(self):
        payload = {"analysis": {"tasks": [{"name": "t", "estimates": [{"role": "qa", "hours": 8}]}],
                                "sizing": "строка вместо объекта", "team_options": ["мусор", 5],
                                "functional_requirements": [None, "x", {"code": "FR-1", "title": "ok"}]}}
        analysis = estimator.normalize_analysis(payload)
        self.assertEqual(analysis["sizing"], {})
        self.assertEqual(analysis["team_options"], [])
        self.assertEqual(analysis["functional_requirements"], [{"code": "FR-1", "title": "ok"}])

    def test_report_contains_sections(self):
        svc = make_service()
        pid = svc.create("Отчётный проект", "Клиент", "Описание с интеграцией API", [])["id"]
        md = svc.run_estimate(pid)["report_markdown"]
        for section in ("## Требования", "## Сайзинг", "## Команда", "## Бюджет", "## Риски", "### Расходы по месяцам"):
            self.assertIn(section, md)
        json.dumps(svc.storage.get_presale(pid)["result"])  # результат сериализуем



class MockChatTests(unittest.TestCase):
    def setUp(self):
        self.svc = make_service()
        self.pid = self.svc.create("Чат-проект", None, "Интеграция с 1С и безопасность данных", [])["id"]

    def test_chat_before_estimate_asks_to_calculate(self):
        self.assertIn("не рассчитана", self.svc.chat(self.pid, "Какие риски?"))

    def test_chat_answers_by_topic(self):
        self.svc.run_estimate(self.pid)
        self.assertIn("Требования могут измениться", self.svc.chat(self.pid, "Какие риски?"))
        self.assertIn("Разработка:", self.svc.chat(self.pid, "Сколько это стоит?"))
        self.assertIn("Модульный монолит", self.svc.chat(self.pid, "Какая архитектура?"))
        self.assertIn("FR-001", self.svc.chat(self.pid, "Какие требования?"))

    def test_chat_unknown_topic_lists_sections(self):
        self.svc.run_estimate(self.pid)
        self.assertIn("Уточните вопрос", self.svc.chat(self.pid, "Привет"))


if __name__ == "__main__":
    unittest.main()
