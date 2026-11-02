# chaai-agent-loop

A LangGraph ReAct harness for terminal assistants, and the small tools they share. Each new assistant needs only a system prompt and its tools: the loop, approval before every tool call, images, saved conversations and memory come with it.

Built by Harleen Serai (Cha.AI) for her own assistants: a poetry submissions assistant, a collaborator for translating South Asian poetry, a bank reconciliation agent and a Punjabi worksheet author.

## What's in it

| Package | What it does |
|---|---|
| `agent_loop` | The harness: `run_agent(...)` starts a terminal conversation with a LangGraph ReAct agent |
| `gmail_tools` | Google sign-in for any API, and reading and sending Gmail (quoted replies and signatures stripped) |
| `smtp_email` | `send_email(...)` through Gmail or iCloud SMTP with an app password: no OAuth, no token expiry |
| `second_opinion` | `ask_claude` and `ask_gpt`: a local-model harness asks a cloud model one hard question |

## Install

```bash
pip install git+https://github.com/harleen/chaai-agent-loop.git
```

Optional extras: `[openai]` for GPT models, `[ollama]` for local models, `[google]` for `gmail_tools`:

```bash
pip install "chaai-agent-loop[ollama,google] @ git+https://github.com/harleen/chaai-agent-loop.git"
```

For development, an editable install from a clone: `pip install -e path/to/chaai-agent-loop`.

Python 3.11+. Anthropic is the default provider (`ANTHROPIC_API_KEY`); OpenAI needs `OPENAI_API_KEY`; Ollama needs a local Ollama running.

## agent_loop

```python
from agent_loop import run_agent
from langchain_core.tools import tool

@tool
def my_tool(query: str) -> str:
    """Does something useful."""
    return f"result for {query}"

run_agent(
    tools=[my_tool],
    system_prompt="You are a helpful assistant. Use my_tool when needed.",
    thread_id="my-session",            # one conversation; give it a natural expiry (e.g. ISO week)
    db_path="checkpoints.db",          # created on first run
    fresh_start_message="Let's go.",   # sent to start a new conversation
    resume_message="Welcome back.",    # sent when resuming one
    verbatim_tool_names=["my_tool"],   # print this tool's output as is, not summarized
    auto_approve_tool_names=[],        # tools that run without asking (read-only ones)
    model="claude-haiku-4-5-20251001", # the default
    model_provider="anthropic",        # or "openai" or "ollama"
    memory_path="memory.db",           # optional: general memory (below)
)
```

### What it does

- **Approval before every tool call.** The graph pauses (`interrupt_before=["tools"]`) and shows the call: Enter runs it; typing anything else cancels it, and the agent reconsiders with what you said. Tools in `auto_approve_tool_names` skip the pause.
- **Images.** Drag one or more files from Finder into the terminal and the model sees the images. HEIC and anything over the API's 5 MB limit become a downscaled JPEG through macOS's `sips`.
- **Saved conversations.** `SqliteSaver` keeps the whole history, by `thread_id`, across restarts.
- **Bounded context.** Only the last `max_context_messages` (default 50) are sent to the model each turn; the full history stays saved, and a built-in `recall_earlier_messages` tool searches back through it. It exists because a thread left open past its rotation once grew to 617 messages and cost about six times as much per call.
- **Clean Ctrl+C.** Tool calls left hanging are closed out, so the database is valid on the next start.
- **Verbatim output.** Named tools print their output directly instead of the model's summary of it.
- **`open_in_browser`**, built in: opens a URL in the default browser, after approval like any tool.
- **Model tiers.** `TOOL_HARNESS_MODEL` (cheap and fast, for tool routing) and `REASONING_MODEL` (for judgement-heavy calls) are kept in one place, to update as new models ship.
- **Standing preferences** (`preferences_path` and `preference_categories`) add a `save_preference` tool writing to a JSON file. Memory is replacing it.

### Memory

`memory_path` gives an assistant a general memory: what holds beyond any one piece of work, such as principles, preferences and facts ("keep refrains in the original", "the reading period opens in September"). Notes about one piece of work belong in that work's own files, kept by the harness, not here.

Five tools are added, auto-approved because they only write locally and never delete:

| Tool | What it does |
|---|---|
| `remember(value, type, key=None, details=None, you_said=False)` | `you_said=True` only when the user asked; then it's `saved`. Anything the model infers itself is `pending`, a suggestion, until approved |
| `correct(key_or_id, value)` | Replaces a memory; the old one is kept as `forgotten`, so a correction can be undone |
| `forget(key_or_id)` | Marks it `forgotten`; nothing is ever deleted |
| `recall(type=None, query=None, status="saved")` | Lists memories, printed in full |
| `approve(id, value=None)` | Keeps a suggestion, reworded if `value` is given |

Types: `fact`, `preference`, `goal`, `habit` and `event`. A `key` makes a correction replace a memory instead of duplicating it. Every saved memory goes into the system prompt at the start, with a line saying how many suggestions are waiting. The table is `observations` in its own SQLite file; columns are added through numbered migrations (SQLite's `user_version`), and anything one assistant needs goes in the `details` JSON first.

### How the approval loop works

```
You type something
  ↓
the agent thinks and decides to call a tool
  ↓
PAUSE: [→ my_tool(query='hello')]
  [Enter] proceed  |  type to redirect:
  ↓ Enter
the tool runs, the agent answers
  ↓
You: (next message)
```

## gmail_tools

```python
from gmail_tools import get_creds, get_service, build_query, fetch_messages, message_full
```

`get_creds(scopes, token_path, credentials_path)` runs Google's desktop OAuth flow for any API and caches the token; each project keeps its own `credentials.json` and token, never committed. `get_service`, `build_query`, `fetch_messages` and `message_text` read Gmail; `message_full` also returns From, To and Cc; `strip_quoted` cuts the quoted thread from a reply and `strip_signature` the sender's signature; `send_email` sends through the Gmail API. Needs the `[google]` extra.

## smtp_email

```python
from pathlib import Path
from smtp_email import send_email

send_email(to="you@example.com", subject="Hello", body_html="<p>Hello</p>",
           from_addr="you@gmail.com", password_path=Path("~/.credentials/gmail_smtp_password").expanduser())
```

For scheduled jobs: an app password in a file, no OAuth and nothing to expire. Gmail by default; `ICLOUD_SMTP_HOST` and `ICLOUD_SMTP_PORT` for iCloud.

## second_opinion

```python
from second_opinion import ask_claude, ask_gpt
run_agent(tools=[..., ask_claude, ask_gpt], model="qwen3:8b", model_provider="ollama", ...)
```

One standalone question to a stronger cloud model, with no shared history or tools, so a local harness can escalate a hard sub-question without moving the whole conversation off the machine. `ask_gpt` needs the `[openai]` extra.

## Tests and evals

```bash
python -m unittest agent_loop.tests.test_memory      # 23 tests, no model, under a second
python -m agent_loop.evals.memory_eval openai gpt-5.5
python -m agent_loop.evals.memory_eval anthropic claude-haiku-4-5
python -m agent_loop.evals.memory_eval ollama qwen3:8b
```

The memory eval gives a real model eleven exchanges with the memory tools ("from now on, keep refrains in the original", a preference only implied, a correction, a forget, approving a suggestion, a note that belongs to the poem in hand, a plain question) and judges what ended up saved, waiting or forgotten. Runs are appended to `agent_loop/evals/results/memory.jsonl`. On 2026-11-02: `gpt-5.5` 11/11, `claude-haiku-4-5` 11/11, `qwen3:8b` 7/11 (it saved an implied preference as though asked, so a local harness needs a firmer prompt before memory is on).

## Notes

Built and used on macOS: the image handling expects paths dragged from Finder and converts with `sips`. Dependencies are pinned with `~=`, so patch releases install on their own and minor ones are upgraded deliberately; LangGraph in particular changes its API between minor versions.

GPL-3.0.
