"""
Does a real model use the memory tools well? One exchange per case, the memory
tools (plus a stand-in add_note for work notes) run for real for up to four
steps, then the store judged: what was saved, what waits, what was forgotten.
PASS is the ideal; "pass" is acceptable (e.g. nothing saved unasked, but no
suggestion either); FAIL is wrong. Uses the harness's own model and key, so it costs a few cents.

    cd ~/Documents/utils
    <venv>/bin/python -m agent_loop.evals.memory_eval openai gpt-5.5            # translation-tools
    <venv>/bin/python -m agent_loop.evals.memory_eval anthropic claude-haiku-4-5 # poetry-tools --cloud
    <venv>/bin/python -m agent_loop.evals.memory_eval ollama qwen3:8b            # poetry-tools default

Results are appended to agent_loop/evals/results/memory.jsonl.
"""

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from agent_loop.memory import MemoryStore, memory_tools

BASE_PROMPT = (
    "You are a translation collaborator helping Harleen Serai translate South Asian poetry "
    "(Punjabi, Hindi, Urdu) into English. We are working on the Bulleh Shah kafi 'Ishq Marvel'. "
    "Use add_note for anything about this poem; it goes in the poem's own Notes.md."
)


@tool
def add_note(note: str) -> str:
    """Add a note about the poem in hand (context, an observation, an open question,
    material for the write-up) to its Notes.md."""
    return "Note added."


# seed: memories in the store before the exchange, as (value, type, key, you_said)
# ok: judged on the outcome: the store after the exchange, and the tools called
def saved(store, type_=None):
    return [r for r in store.recall(type_) if r["status"] == "saved"]


def pending(store):
    return store.recall(status="pending")


def called(calls, name):
    return any(c["name"] == name for c in calls)


MEMORY_TOOLS = {"remember", "correct", "forget", "approve"}

CASES = [
    {"name": "asked: a principle", "say": "From now on, keep refrains in the original.",
     "ok": lambda s, c: len(saved(s, "preference")) == 1},
    {"name": "asked: a fact", "say": "Remember that Adroit's reading period opens in September.",
     "ok": lambda s, c: len(saved(s, "fact")) == 1},
    {"name": "asked: always", "say": "Always use American spelling in my translations.",
     "ok": lambda s, c: len(saved(s, "preference")) == 1},
    {"name": "asked, indirectly",
     "say": "Let's go with 'longing' again. Like last time, plain present-day words just work better for me than the old-fashioned ones. Can you note that preference so you remember it?",
     "ok": lambda s, c: len(saved(s, "preference")) == 1},
    {"name": "inferred, not asked",
     "say": "'Longing' again, I think. Plain words keep working better for me than the old-fashioned ones, like with the Faiz poem.",
     # nothing saved she didn't ask for; a suggestion is the ideal
     "ok": lambda s, c: not saved(s), "ideal": lambda s, c: len(pending(s)) == 1},
    {"name": "correct", "seed": [("Keep refrains untranslated", "preference", "refrains", True)],
     "say": "No, about refrains: keep the refrain in the original, but gloss it in the commentary.",
     "ok": lambda s, c: len(saved(s)) == 1 and "gloss" in saved(s)[0]["value"].lower()},
    {"name": "forget", "seed": [("Adroit's reading period opens in September", "fact", "adroit_reading_period", True)],
     "say": "Forget what I told you about Adroit's reading period.",
     "ok": lambda s, c: not saved(s)},
    {"name": "approve a suggestion", "seed": [("Prefers plain, present-day English over archaic register", "preference", "register", False)],
     "say": "Yes, keep that suggestion about plain English. It's right.",
     "ok": lambda s, c: len(saved(s)) == 1 and not pending(s)},
    {"name": "recall", "seed": [("Gloss words with no English equivalent", "preference", "glosses", True)],
     "say": "What do you remember about how I like glosses?",
     "ok": lambda s, c: len(saved(s)) == 1 and not any(x["name"] in MEMORY_TOOLS for x in c)},
    {"name": "poem note, not memory",
     "say": "Add a note: in this stanza 'birha' echoes the separation in Heer-Ranjha. I want it for the write-up.",
     "ok": lambda s, c: called(c, "add_note") and not saved(s) and not pending(s)},
    {"name": "no memory", "say": "What's a good English word for 'birha' here?",
     "ok": lambda s, c: not saved(s) and not pending(s)},
]


def make_model(provider: str, model: str):
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=0)
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, temperature=0, num_predict=8192)
    raise SystemExit(f"unknown provider {provider}")


def run_case(model, case) -> tuple[bool, bool, str]:
    """Up to four tool steps, the tools run for real; judged on the store afterwards."""
    with tempfile.TemporaryDirectory() as d:
        store = MemoryStore(Path(d) / "memory.db")
        for value, type_, key, you_said in case.get("seed", []):
            store.remember(value, type_, key=key, you_said=you_said)
        tools = {t.name: t for t in memory_tools(store) + [add_note]}
        bound = model.bind_tools(list(tools.values()))
        messages = [SystemMessage(BASE_PROMPT + store.prompt_block()), HumanMessage(case["say"])]
        calls = []
        for _ in range(4):
            reply = bound.invoke(messages)
            messages.append(reply)
            if not reply.tool_calls:
                break
            for tc in reply.tool_calls:
                calls.append(tc)
                messages.append(ToolMessage(str(tools[tc["name"]].invoke(tc["args"])), tool_call_id=tc["id"]))
        ok = case["ok"](store, calls)
        ideal = case.get("ideal", case["ok"])(store, calls)
        store.db.close()
    got = " → ".join(f"{c['name']}({json.dumps(c['args'], ensure_ascii=False)[:90]})" for c in calls) or "no tool call"
    return ok, ideal, got


def main() -> None:
    provider, model_name = sys.argv[1], sys.argv[2]
    model = make_model(provider, model_name)
    results, passed = [], 0
    for case in CASES:
        ok, ideal, got = run_case(model, case)
        passed += ok
        mark = "PASS" if ok and ideal else ("pass" if ok else "FAIL")
        print(f"{mark}  {case['name']:24} {got}")
        results.append({"case": case["name"], "ok": ok, "ideal": ideal, "got": got})
    print(f"\n{provider} {model_name}: {passed}/{len(CASES)}")
    out = Path(__file__).parent / "results" / "memory.jsonl"
    out.parent.mkdir(exist_ok=True)
    with out.open("a") as f:
        f.write(json.dumps({"at": datetime.now().isoformat(timespec="seconds"), "provider": provider,
                            "model": model_name, "passed": passed, "of": len(CASES), "results": results}) + "\n")


if __name__ == "__main__":
    main()
