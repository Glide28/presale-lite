"""Хранилище на SQLite (стандартная библиотека, без отдельного сервера БД)."""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS presales (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    customer TEXT,
    description TEXT NOT NULL DEFAULT '',
    outputs TEXT NOT NULL DEFAULT '[]',
    rates TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'draft',
    result TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    presale_id TEXT NOT NULL REFERENCES presales(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS answers (
    id TEXT PRIMARY KEY,
    presale_id TEXT NOT NULL REFERENCES presales(id) ON DELETE CASCADE,
    question TEXT NOT NULL,
    answer TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    presale_id TEXT NOT NULL REFERENCES presales(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _id() -> str:
    return uuid.uuid4().hex


class NotFound(LookupError):
    pass


class Storage:
    def __init__(self, path: str = "presale.db") -> None:
        self._path = path
        # для :memory: держим одно соединение, иначе каждая операция видит пустую БД
        self._shared = sqlite3.connect(path, check_same_thread=False) if path == ":memory:" else None
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = self._shared or sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            if self._shared is None:
                conn.close()

    # --- presales ---
    def create_presale(self, title: str, customer: str | None, description: str,
                       outputs: list[str]) -> dict:
        pid = _id()
        with self._conn() as c:
            c.execute(
                "INSERT INTO presales(id, title, customer, description, outputs, created_at)"
                " VALUES (?,?,?,?,?,?)",
                (pid, title, customer, description, json.dumps(outputs, ensure_ascii=False), _now()),
            )
        return self.get_presale(pid)

    def list_presales(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM presales ORDER BY created_at DESC").fetchall()
        return [self._presale(r) for r in rows]

    def get_presale(self, pid: str) -> dict:
        with self._conn() as c:
            row = c.execute("SELECT * FROM presales WHERE id=?", (pid,)).fetchone()
        if row is None:
            raise NotFound(pid)
        return self._presale(row)

    def set_rates(self, pid: str, rates: list[dict]) -> None:
        self.get_presale(pid)
        with self._conn() as c:
            c.execute("UPDATE presales SET rates=? WHERE id=?",
                      (json.dumps(rates, ensure_ascii=False), pid))

    def save_result(self, pid: str, result: dict) -> None:
        self.get_presale(pid)
        with self._conn() as c:
            c.execute("UPDATE presales SET result=?, status='estimated' WHERE id=?",
                      (json.dumps(result, ensure_ascii=False), pid))

    def delete_presale(self, pid: str) -> None:
        self.get_presale(pid)
        with self._conn() as c:
            c.execute("DELETE FROM presales WHERE id=?", (pid,))

    @staticmethod
    def _presale(row: sqlite3.Row) -> dict:
        data = dict(row)
        data["outputs"] = json.loads(data["outputs"])
        data["rates"] = json.loads(data["rates"])
        data["result"] = json.loads(data["result"]) if data["result"] else None
        return data

    # --- documents ---
    def add_document(self, pid: str, filename: str, text: str) -> dict:
        self.get_presale(pid)
        did = _id()
        with self._conn() as c:
            c.execute("INSERT INTO documents(id, presale_id, filename, text, created_at)"
                      " VALUES (?,?,?,?,?)", (did, pid, filename, text, _now()))
        return {"id": did, "filename": filename, "chars": len(text)}

    def list_documents(self, pid: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT id, filename, text FROM documents WHERE presale_id=?"
                             " ORDER BY created_at", (pid,)).fetchall()
        return [dict(r) for r in rows]

    # --- answers ---
    def set_answers(self, pid: str, qa: list[tuple[str, str]]) -> None:
        self.get_presale(pid)
        with self._conn() as c:
            c.execute("DELETE FROM answers WHERE presale_id=?", (pid,))
            c.executemany("INSERT INTO answers(id, presale_id, question, answer) VALUES (?,?,?,?)",
                          [(_id(), pid, q, a) for q, a in qa])

    def list_answers(self, pid: str) -> list[tuple[str, str]]:
        with self._conn() as c:
            rows = c.execute("SELECT question, answer FROM answers WHERE presale_id=?", (pid,)).fetchall()
        return [(r["question"], r["answer"]) for r in rows]

    # --- chat ---
    def add_message(self, pid: str, role: str, content: str) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO messages(id, presale_id, role, content, created_at)"
                      " VALUES (?,?,?,?,?)", (_id(), pid, role, content, _now()))

    def list_messages(self, pid: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT role, content, created_at FROM messages WHERE presale_id=?"
                             " ORDER BY created_at, rowid", (pid,)).fetchall()
        return [dict(r) for r in rows]
