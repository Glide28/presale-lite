"""Диагностика подключения к модели: python diagnose_llm.py [--fast]

Показывает, на каком шаге проблема: сеть/VPN, адрес, ключ, баланс или скорость модели.
--fast  отключает «размышления» модели (LLM_REASONING_EFFORT=none) для проверки скорости.
Ключ в вывод не попадает.
"""
from __future__ import annotations

import dataclasses
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from app.config import Settings
from app.llm import LLMError, OpenAICompatibleLLM


def step(title: str) -> None:
    print(f"\n=== {title}")


def main() -> int:
    settings = Settings.from_env()
    if "--fast" in sys.argv:
        settings = dataclasses.replace(settings, llm_reasoning_effort="none")

    step("1. Настройки")
    key = settings.llm_api_key
    print(f"провайдер:  {settings.llm_provider}")
    print(f"адрес:      {settings.llm_base_url}")
    print(f"модель:     {settings.llm_model}")
    print(f"таймаут:    {settings.llm_timeout} с, reasoning: {settings.llm_reasoning_effort or 'по умолчанию'}")
    print(f"ключ:       {'задан, длина ' + str(len(key)) + ', начало ' + key[:9] + '...' if key else 'НЕ ЗАДАН'}")
    if settings.llm_provider != "openai" or not key:
        print("\nПроблема: LLM_PROVIDER должен быть openai, а LLM_API_KEY не пустым. Проверьте файл .env "
              "в папке проекта (имя ровно .env).")
        return 1

    parsed = urllib.parse.urlparse(settings.llm_base_url)
    host, port = parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)

    step("2. Прокси системы")
    proxies = urllib.request.getproxies()
    print(proxies if proxies else "не настроены (Python подключается напрямую)")

    step(f"3. DNS и TCP до {host}:{port}")
    try:
        started = time.time()
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        print(f"DNS: {len(infos)} адрес(ов) за {time.time() - started:.1f} с")
    except OSError as exc:
        print(f"DNS не сработал: {exc}")
        return 1
    reachable = False
    for family, _, _, _, addr in infos:
        name = "IPv6" if family == socket.AF_INET6 else "IPv4"
        started = time.time()
        try:
            with socket.create_connection(addr[:2], timeout=8):
                print(f"  {name} {addr[0]}: соединение OK ({time.time() - started:.1f} с)")
                reachable = True
        except OSError as exc:
            print(f"  {name} {addr[0]}: НЕТ соединения ({exc})")
    if not reachable:
        print("\nПроблема: Python не может подключиться к серверу. Расширение VPN в браузере не работает "
              "для Python. Включите системный VPN (всё устройство) или настройте прокси и перезапустите.")
        return 1

    step("4. Запрос списка моделей (без ключа)")
    try:
        started = time.time()
        with urllib.request.urlopen(f"{settings.llm_base_url}/models", timeout=20) as resp:
            print(f"HTTP {resp.status} за {time.time() - started:.1f} с")
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"Не удалось: {exc}")

    step("5. Тестовый запрос к модели с вашим ключом")
    llm = OpenAICompatibleLLM(settings)
    started = time.time()
    try:
        answer = llm.complete("Отвечай одним словом.", "Скажи слово: готово", json_mode=False)
    except LLMError as exc:
        print(f"ОШИБКА за {time.time() - started:.1f} с: {exc}")
        return 1
    print(f"Ответ за {time.time() - started:.1f} с: {answer.strip()[:200]}")

    step("Итог")
    print("Подключение работает. Если оценка всё равно идёт долго, запустите: python diagnose_llm.py --fast "
          "и при ускорении добавьте в .env строку LLM_REASONING_EFFORT=none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
