"""
run_agent with memory_path: the memory tools are added and auto-approved,
recall prints verbatim and the memory block ends the system prompt. The
conversation loop is stubbed out, so no model is called.

    python -m unittest agent_loop.tests.test_run_agent_memory
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import agent_loop
from agent_loop.memory import MEMORY_TOOL_NAMES, MemoryStore


class RunAgentMemoryTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.memory = Path(self.dir.name) / "memory.db"
        self.seen = {}

    def tearDown(self):
        self.dir.cleanup()

    def run_agent(self, **kwargs):
        """Builds the agent as run_agent does, recording what it was given."""
        real_create = agent_loop.create_react_agent

        def spy_create(model, tools, **kw):
            self.seen["tools"] = [t.name for t in tools]
            content = kw["prompt"].content  # Anthropic: a list of blocks, for prompt caching
            self.seen["prompt"] = content if isinstance(content, str) else "".join(b["text"] for b in content)
            return real_create(model, tools=tools, **kw)

        def fake_loop(graph, config, **kw):
            self.seen["auto_approve"] = kw["auto_approve_tool_names"]
            self.seen["verbatim"] = kw["verbatim_tool_names"]

        # A placeholder key: the model is built but never called.
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", "test")}), \
             mock.patch.object(agent_loop, "create_react_agent", spy_create), \
             mock.patch.object(agent_loop, "_conversation_loop", fake_loop):
            agent_loop.run_agent(tools=[], system_prompt="Test assistant.", thread_id="t1",
                                 db_path=str(Path(self.dir.name) / "checkpoints.db"), **kwargs)

    def test_memory_tools_are_added_and_auto_approved(self):
        self.run_agent(memory_path=self.memory)
        self.assertTrue(MEMORY_TOOL_NAMES <= set(self.seen["tools"]))
        self.assertTrue(MEMORY_TOOL_NAMES <= self.seen["auto_approve"])
        self.assertIn("recall", self.seen["verbatim"])

    def test_saved_memories_end_the_prompt(self):
        store = MemoryStore(self.memory)
        store.remember("Keep refrains in the original", "preference", key="refrains", you_said=True)
        store.remember("Prefers plain English", "preference", key="register")
        store.db.close()
        self.run_agent(memory_path=self.memory)
        prompt = self.seen["prompt"]
        self.assertTrue(prompt.startswith("Test assistant."))
        self.assertIn("[refrains] Keep refrains in the original", prompt)
        self.assertNotIn("Prefers plain English", prompt)
        self.assertIn("1 suggested memories are waiting", prompt)

    def test_without_memory_path_nothing_changes(self):
        self.run_agent()
        self.assertFalse(MEMORY_TOOL_NAMES & set(self.seen["tools"]))
        self.assertNotIn("What you remember", self.seen["prompt"])
        self.assertFalse(self.memory.exists())

    def test_the_harness_verbatim_tools_are_kept(self):
        self.run_agent(memory_path=self.memory, verbatim_tool_names=["show_block"])
        self.assertEqual(self.seen["verbatim"], ["show_block", "recall"])


if __name__ == "__main__":
    unittest.main()
