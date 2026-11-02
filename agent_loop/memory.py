"""
General memory for an assistant: what it has learned that holds beyond any one
piece of work (principles, preferences, facts). Work material (a poem's notes
and decisions) stays in the work's own files, kept by the harness.


    store = MemoryStore("memory.db")
    tools = memory_tools(store, thread_id="faiz-god-truthful")

Rows are never deleted: forget and correct mark the old row `forgotten`, so
any mistake can be undone. Only `saved` rows reach the prompt; what the model
infers waits as `pending` until the user approves it.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.tools import tool

TYPES = ("fact", "preference", "goal", "habit", "event")
STATUSES = ("pending", "saved", "forgotten", "expired")

# Each entry moves the database up one version (SQLite's user_version).
# Add new columns as a new entry at the end; never edit an old one.
MIGRATIONS = [
    """
    CREATE TABLE observations (
        id             INTEGER PRIMARY KEY,
        type           TEXT NOT NULL,
        key            TEXT,
        value          TEXT NOT NULL,
        status         TEXT NOT NULL,
        confidence     REAL,
        times_observed INTEGER NOT NULL DEFAULT 1,
        source         TEXT,
        created_at     TEXT NOT NULL,
        updated_at     TEXT NOT NULL,
        expires_at     TEXT,
        details        TEXT
    );
    CREATE INDEX observations_status ON observations(status);
    CREATE INDEX observations_key ON observations(key);
    """,
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MemoryStore:
    def __init__(self, path: "str | Path"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self._migrate()

    def _migrate(self) -> None:
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        for i, sql in enumerate(MIGRATIONS[version:], start=version + 1):
            with self.db:
                self.db.executescript(sql)
                self.db.execute(f"PRAGMA user_version = {i}")

    # ── lookups ──────────────────────────────────────────────────────────────

    def _find(self, key_or_id: str, statuses=("saved",)) -> "sqlite3.Row | None":
        """A row by id, else the newest row with that key, among `statuses`."""
        marks = ",".join("?" * len(statuses))
        if str(key_or_id).isdigit():
            row = self.db.execute(
                f"SELECT * FROM observations WHERE id = ? AND status IN ({marks})",
                (int(key_or_id), *statuses),
            ).fetchone()
            if row:
                return row
        return self.db.execute(
            f"SELECT * FROM observations WHERE key = ? AND status IN ({marks}) ORDER BY id DESC LIMIT 1",
            (key_or_id, *statuses),
        ).fetchone()

    def _set_status(self, row_id: int, status: str) -> None:
        self.db.execute("UPDATE observations SET status = ?, updated_at = ? WHERE id = ?", (status, _now(), row_id))

    def _retire_key(self, key: "str | None", keep_id: int) -> None:
        """A saved memory replaces the older saved one with the same key."""
        if key:
            self.db.execute(
                "UPDATE observations SET status = 'forgotten', updated_at = ? WHERE key = ? AND status = 'saved' AND id != ?",
                (_now(), key, keep_id),
            )

    # ── the verbs ────────────────────────────────────────────────────────────

    def remember(self, value: str, type: str, key=None, details=None, you_said=False, source_details=None) -> dict:
        if type not in TYPES:
            raise ValueError(f"type must be one of {TYPES}")
        value = value.strip()
        status = "saved" if you_said else "pending"
        if key:
            same = self.db.execute(
                "SELECT * FROM observations WHERE key = ? AND value = ? AND status IN ('saved', 'pending')", (key, value)
            ).fetchone()
            if same:
                with self.db:
                    self.db.execute(
                        "UPDATE observations SET times_observed = times_observed + 1, updated_at = ? WHERE id = ?",
                        (_now(), same["id"]),
                    )
                return {"id": same["id"], "status": same["status"], "repeat": True}
        all_details = {**(details or {}), **(source_details or {})}
        with self.db:
            cur = self.db.execute(
                "INSERT INTO observations (type, key, value, status, source, created_at, updated_at, details) VALUES (?,?,?,?,?,?,?,?)",
                (type, key, value, status, "you said" if you_said else "inferred", _now(), _now(),
                 json.dumps(all_details) if all_details else None),
            )
            if status == "saved":
                self._retire_key(key, cur.lastrowid)
        return {"id": cur.lastrowid, "status": status, "repeat": False}

    def correct(self, key_or_id: str, value: str) -> "dict | None":
        old = self._find(key_or_id, ("saved", "pending"))
        if not old:
            return None
        with self.db:
            cur = self.db.execute(
                "INSERT INTO observations (type, key, value, status, source, created_at, updated_at, details) VALUES (?,?,?,?,?,?,?,?)",
                (old["type"], old["key"], value.strip(), "saved", "you said", _now(), _now(),
                 json.dumps({**json.loads(old["details"] or "{}"), "corrects": old["id"]})),
            )
            self._set_status(old["id"], "forgotten")
            self._retire_key(old["key"], cur.lastrowid)
        return {"id": cur.lastrowid, "replaced": old["id"]}

    def forget(self, key_or_id: str) -> "int | None":
        row = self._find(key_or_id, ("saved", "pending"))
        if not row:
            return None
        with self.db:
            self._set_status(row["id"], "forgotten")
        return row["id"]

    def approve(self, row_id: int, value=None) -> "dict | None":
        row = self._find(str(row_id), ("pending",))
        if not row:
            return None
        with self.db:
            if value:
                self.db.execute("UPDATE observations SET value = ? WHERE id = ?", (value.strip(), row["id"]))
            self._set_status(row["id"], "saved")
            self._retire_key(row["key"], row["id"])
        return {"id": row["id"]}

    def recall(self, type=None, query=None, status="saved") -> list:
        sql, args = "SELECT * FROM observations WHERE status = ?", [status]
        if type:
            sql += " AND type = ?"
            args.append(type)
        if query:
            sql += " AND (value LIKE ? OR key LIKE ?)"
            args += [f"%{query}%", f"%{query}%"]
        return self.db.execute(sql + " ORDER BY type, id", args).fetchall()

    # ── the prompt ───────────────────────────────────────────────────────────

    def prompt_block(self) -> str:
        """Every saved memory, for the system prompt, plus how many are pending."""
        rows = self.recall()
        pending = len(self.recall(status="pending"))
        lines = ["\n\n## What you remember (general memory, across all work)"]
        if rows:
            current = None
            for r in rows:
                if r["type"] != current:
                    current = r["type"]
                    lines.append(f"\n{current.capitalize()}s:")
                key = f"[{r['key']}] " if r["key"] else ""
                lines.append(f"- {key}{r['value']} (#{r['id']})")
        else:
            lines.append("Nothing saved yet.")
        if pending:
            lines.append(f"\n{pending} suggested memories are waiting for the user's approval (recall status='pending').")
        lines.append(
            "\nMemory is for what holds beyond the work in hand. Notes and decisions about one "
            "piece of work go in that work's own files, not here. Use remember with "
            "you_said=true only when the user asked you to remember it or corrected you; "
            "anything you infer yourself is a suggestion (you_said=false) that waits for them."
        )
        return "\n".join(lines)


def _format(rows) -> str:
    if not rows:
        return "(nothing)"
    out = []
    for r in rows:
        key = f"[{r['key']}] " if r["key"] else ""
        out.append(f"#{r['id']} {r['type']} {key}{r['value']}  ({r['status']}, {r['source']}, {r['updated_at'][:10]})")
    return "\n".join(out)


def memory_tools(store: MemoryStore, thread_id: "str | None" = None) -> list:
    """The harness-level verbs, bound to one store."""
    source_details = {"thread": thread_id} if thread_id else None

    @tool
    def remember(value: str, type: str, key: "str | None" = None, details: "dict | None" = None, you_said: bool = False) -> str:
        """Remember something general that holds beyond the work in hand: a principle,
        preference, fact, goal or event. Not for notes or decisions about one piece of
        work; those go in the work's own files.

        value: the memory, written to read clearly later, out of this conversation's context.
        type: one of fact, preference, goal, habit, event.
        key: a short stable name (e.g. "refrains", "reading_period_adroit") so a later
          correction replaces this memory instead of duplicating it.
        details: optional extra fields.
        you_said: true ONLY when the user asked you to remember it ("remember…", "from now
          on…", "always…"); then it is saved. Leave false for anything you noticed or
          inferred yourself: it becomes a suggestion they approve later.
        """
        try:
            r = store.remember(value, type, key, details, you_said, source_details)
        except ValueError as e:
            return str(e)
        if r["repeat"]:
            return f"Already remembered (#{r['id']}, {r['status']}); counted it again."
        if r["status"] == "saved":
            return f"Saved (#{r['id']})."
        return f"Suggested (#{r['id']}): waiting for the user's approval."

    @tool
    def correct(key_or_id: str, value: str) -> str:
        """Replace a memory the user has corrected ("no, it's…"). The old version is kept
        as forgotten, so the correction can be undone. key_or_id: its key or #id."""
        r = store.correct(key_or_id, value)
        if not r:
            return f"No memory found for '{key_or_id}'. Use recall to find it, or remember it new."
        return f"Corrected: #{r['id']} replaces #{r['replaced']}."

    @tool
    def forget(key_or_id: str) -> str:
        """Forget a memory (or drop a suggestion) when the user asks. Nothing is deleted:
        it is marked forgotten and stops reaching the prompt. key_or_id: its key or #id."""
        row_id = store.forget(key_or_id)
        if row_id is None:
            return f"No memory found for '{key_or_id}'."
        return f"Forgotten (#{row_id})."

    @tool
    def recall(type: "str | None" = None, query: "str | None" = None, status: str = "saved") -> str:
        """List memories. status: saved (default), pending (suggestions waiting for
        the user), forgotten or expired. type and query narrow it. Output is shown to
        the user verbatim."""
        if status not in STATUSES:
            return f"status must be one of {STATUSES}"
        return _format(store.recall(type, query, status))

    @tool
    def approve(id: int, value: "str | None" = None) -> str:
        """Keep a suggested memory the user agreed to, edited to `value` if they changed
        the wording. To drop a suggestion, use forget instead."""
        r = store.approve(id, value)
        if not r:
            return f"No suggestion #{id} waiting."
        return f"Saved (#{r['id']})."

    return [remember, correct, forget, recall, approve]


MEMORY_TOOL_NAMES = {"remember", "correct", "forget", "recall", "approve"}
VERBATIM_MEMORY_TOOLS = ["recall"]
