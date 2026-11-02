"""
Tests for agent_loop/memory.py: the store and the tools the harness adds.
No model is involved.

    cd ~/Documents/utils
    <any harness venv>/bin/python -m unittest agent_loop.tests.test_memory
"""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from agent_loop.memory import MIGRATIONS, MemoryStore, memory_tools


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "memory.db"
        self.store = MemoryStore(self.path)

    def tearDown(self):
        self.store.db.close()
        self.dir.cleanup()

    def status(self, row_id):
        return self.store.db.execute("SELECT status FROM observations WHERE id = ?", (row_id,)).fetchone()[0]

    # ── remember ─────────────────────────────────────────────────────────────

    def test_what_she_asked_for_is_saved(self):
        r = self.store.remember("Keep refrains in the original", "preference", key="refrains", you_said=True)
        self.assertEqual(r["status"], "saved")
        self.assertEqual(self.store.recall()[0]["source"], "you said")

    def test_what_the_model_infers_waits(self):
        r = self.store.remember("Prefers plain English over archaic register", "preference", key="register")
        self.assertEqual(r["status"], "pending")
        self.assertEqual(self.store.recall(), [])
        self.assertEqual(self.store.recall(status="pending")[0]["source"], "inferred")

    def test_a_repeat_is_counted_not_duplicated(self):
        first = self.store.remember("Keep refrains in the original", "preference", key="refrains", you_said=True)
        again = self.store.remember("Keep refrains in the original", "preference", key="refrains", you_said=True)
        self.assertTrue(again["repeat"])
        self.assertEqual(again["id"], first["id"])
        row = self.store.recall()[0]
        self.assertEqual(row["times_observed"], 2)
        self.assertEqual(len(self.store.recall()), 1)

    def test_a_bad_type_is_refused(self):
        with self.assertRaises(ValueError):
            self.store.remember("x", "opinion")

    def test_a_new_value_for_a_saved_key_replaces_the_old(self):
        old = self.store.remember("Gloss nothing", "preference", key="glosses", you_said=True)
        new = self.store.remember("Gloss words with no English equivalent", "preference", key="glosses", you_said=True)
        self.assertEqual(self.status(old["id"]), "forgotten")
        self.assertEqual([r["id"] for r in self.store.recall()], [new["id"]])

    def test_a_suggestion_does_not_replace_a_saved_memory(self):
        saved = self.store.remember("Gloss nothing", "preference", key="glosses", you_said=True)
        self.store.remember("Gloss every Sufi term", "preference", key="glosses")
        self.assertEqual(self.status(saved["id"]), "saved")

    def test_details_and_thread_are_kept(self):
        self.store.remember("Adroit reads in September", "fact", details={"journal": "Adroit"},
                            you_said=True, source_details={"thread": "faiz-god-truthful"})
        details = json.loads(self.store.recall()[0]["details"])
        self.assertEqual(details, {"journal": "Adroit", "thread": "faiz-god-truthful"})

    # ── correct, forget, approve ────────────────────────────────────────────

    def test_correct_keeps_the_old_version_as_forgotten(self):
        old = self.store.remember("Keep refrains untranslated", "preference", key="refrains", you_said=True)
        r = self.store.correct("refrains", "Keep the refrain in the original, with a gloss")
        self.assertEqual(r["replaced"], old["id"])
        self.assertEqual(self.status(old["id"]), "forgotten")
        new = self.store.recall()[0]
        self.assertEqual(new["value"], "Keep the refrain in the original, with a gloss")
        self.assertEqual(json.loads(new["details"])["corrects"], old["id"])

    def test_correct_by_id_and_of_a_suggestion(self):
        s = self.store.remember("Prefers archaic register", "preference", key="register")
        r = self.store.correct(str(s["id"]), "Prefers plain, present-day English")
        self.assertEqual(self.status(s["id"]), "forgotten")
        self.assertEqual(self.status(r["id"]), "saved")

    def test_correct_of_nothing_says_so(self):
        self.assertIsNone(self.store.correct("no-such-key", "x"))

    def test_forget_marks_and_never_deletes(self):
        r = self.store.remember("Book club meets on Tuesdays", "fact", key="book_club", you_said=True)
        self.assertEqual(self.store.forget("book_club"), r["id"])
        self.assertEqual(self.store.recall(), [])
        self.assertEqual(self.store.recall(status="forgotten")[0]["id"], r["id"])
        self.assertIsNone(self.store.forget("book_club"))

    def test_approve_saves_with_her_wording(self):
        s = self.store.remember("Prefers plain English", "preference", key="register")
        self.store.approve(s["id"], "Prefer plain, present-day English over archaic or biblical register")
        row = self.store.recall()[0]
        self.assertEqual(row["status"], "saved")
        self.assertEqual(row["value"], "Prefer plain, present-day English over archaic or biblical register")

    def test_approve_replaces_the_saved_memory_with_that_key(self):
        old = self.store.remember("Gloss nothing", "preference", key="glosses", you_said=True)
        s = self.store.remember("Gloss words with no English equivalent", "preference", key="glosses")
        self.store.approve(s["id"])
        self.assertEqual(self.status(old["id"]), "forgotten")
        self.assertEqual([r["id"] for r in self.store.recall()], [s["id"]])

    def test_approve_only_takes_a_suggestion(self):
        saved = self.store.remember("Gloss nothing", "preference", you_said=True)
        self.assertIsNone(self.store.approve(saved["id"]))
        self.assertIsNone(self.store.approve(999))

    # ── recall ───────────────────────────────────────────────────────────────

    def test_recall_narrows_by_type_and_query(self):
        self.store.remember("Keep refrains in the original", "preference", key="refrains", you_said=True)
        self.store.remember("Adroit reads in September", "fact", key="adroit", you_said=True)
        self.assertEqual([r["key"] for r in self.store.recall(type="fact")], ["adroit"])
        self.assertEqual([r["key"] for r in self.store.recall(query="refrain")], ["refrains"])
        self.assertEqual([r["key"] for r in self.store.recall(query="adroit")], ["adroit"])

    # ── the prompt ───────────────────────────────────────────────────────────

    def test_prompt_has_saved_memories_and_the_pending_count(self):
        self.store.remember("Keep refrains in the original", "preference", key="refrains", you_said=True)
        self.store.remember("Prefers plain English", "preference", key="register")
        block = self.store.prompt_block()
        self.assertIn("[refrains] Keep refrains in the original", block)
        self.assertNotIn("Prefers plain English", block)
        self.assertIn("1 suggested memories are waiting", block)

    def test_prompt_with_nothing_saved(self):
        self.assertIn("Nothing saved yet.", self.store.prompt_block())

    # ── migrations ───────────────────────────────────────────────────────────

    def test_reopening_does_not_rerun_migrations(self):
        self.store.remember("Keep refrains in the original", "preference", you_said=True)
        self.store.db.close()
        self.store = MemoryStore(self.path)
        self.assertEqual(self.store.db.execute("PRAGMA user_version").fetchone()[0], len(MIGRATIONS))
        self.assertEqual(len(self.store.recall()), 1)

    def test_an_older_database_is_brought_up_to_date(self):
        self.store.db.close()
        self.path.unlink()
        db = sqlite3.connect(self.path)
        db.executescript(MIGRATIONS[0])
        db.execute("PRAGMA user_version = 1")
        db.close()
        self.store = MemoryStore(self.path)
        self.assertEqual(self.store.db.execute("PRAGMA user_version").fetchone()[0], len(MIGRATIONS))


class ToolsTest(unittest.TestCase):
    """The tools as the model calls them: their names and what they return."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.dir.name) / "memory.db")
        self.tools = {t.name: t for t in memory_tools(self.store, thread_id="bulleh-shah-ishq-marvel")}

    def tearDown(self):
        self.store.db.close()
        self.dir.cleanup()

    def call(self, name, **kwargs):
        return self.tools[name].invoke(kwargs)

    def test_the_five_verbs(self):
        self.assertEqual(sorted(self.tools), ["approve", "correct", "forget", "recall", "remember"])

    def test_a_conversation_round_trip(self):
        self.assertEqual(self.call("remember", value="Keep refrains in the original", type="preference",
                                   key="refrains", you_said=True), "Saved (#1).")
        self.assertIn("waiting for the user's approval",
                      self.call("remember", value="Prefers plain English", type="preference", key="register"))
        self.assertIn("#2 preference [register] Prefers plain English", self.call("recall", status="pending"))
        self.assertEqual(self.call("approve", id=2), "Saved (#2).")
        self.assertEqual(self.call("correct", key_or_id="refrains", value="Gloss the refrain"),
                         "Corrected: #3 replaces #1.")
        self.assertEqual(self.call("forget", key_or_id="register"), "Forgotten (#2).")
        saved = self.call("recall")
        self.assertIn("Gloss the refrain", saved)
        self.assertNotIn("Prefers plain English", saved)

    def test_the_thread_is_recorded(self):
        self.call("remember", value="Adroit reads in September", type="fact", you_said=True)
        self.assertEqual(json.loads(self.store.recall()[0]["details"])["thread"], "bulleh-shah-ishq-marvel")

    def test_mistakes_come_back_as_messages(self):
        self.assertIn("type must be one of", self.call("remember", value="x", type="opinion"))
        self.assertIn("No memory found", self.call("correct", key_or_id="nothing", value="x"))
        self.assertIn("No memory found", self.call("forget", key_or_id="nothing"))
        self.assertIn("No suggestion #9", self.call("approve", id=9))
        self.assertIn("status must be one of", self.call("recall", status="deleted"))
        self.assertEqual(self.call("recall"), "(nothing)")


if __name__ == "__main__":
    unittest.main()
