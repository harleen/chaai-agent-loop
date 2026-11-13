# utils (chaai-agent-loop) — Status

_Last updated: 2026-11-13_

The shared harness under the assistants (`agent_loop`), with `gmail_tools`, `smtp_email` and `second_opinion`. Memory work for `agent_loop` is tracked in `my-chaai/STATUS.md`.

Status legend: `[ ]` not started · `[~]` in progress · `[x]` complete

---

## Build

**Photos kept out of the checkpoint DB** (2026-11-10). PFC's `reconciliation.db` had reached 5.4 GB: every checkpoint saved the whole conversation, photos included, twice (the trim hook's `llm_input_messages` too), and `create_react_agent`'s default v2 gave each queued tool call its own copy (7 calls made one 149 MB row). Now `create_react_agent(version="v1")`; `agent_loop/images.py` resizes a dragged-in photo to 1568 px and stores it once beside the DB (`<db stem>-images/<hash>.jpg`), history holds a reference, and the `prompt` function (which replaced `pre_model_hook`) trims and adds the images back just before the model call, so nothing it builds is saved. Checked with a stand-in model on a real receipt: the model got the image on every call in the turn, the largest checkpoint was 2.2 KB.

Known break: a conversation started before this change can't resume a paused tool call (LangGraph pauses again on every Enter; PFC's 2026-09 thread). New conversations are fine.

- [ ] Harleen: the first live run (PFC 2026-10 or a new translation): Claude reads a dragged-in receipt, a `-images` folder appears
- [ ] An old-layout check: a thread saved by the old graph (its channels include `branch:to:pre_model_hook`) starts fresh with a one-line notice, instead of looping
- [ ] Move `create_react_agent` to `langchain.agents.create_agent` (deprecated in LangGraph 1.0, removed in 2.0)
