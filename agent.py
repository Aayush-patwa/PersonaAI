# agent.py - The brain of JARVIS (tool-calling agent)
#
# How tasks stay in sync with the UI:
#   The LLM never "pretends" to manage tasks. It calls real tools (add_task, list_tasks, ...)
#   that read/write the same SQLite database the Tasks panel uses, and it only reports what
#   the tool results say actually happened.
import os
import re
import sys
import json
import time
import datetime
import traceback

from dotenv import load_dotenv
from groq import Groq

try:
    from ddgs import DDGS                # current package name
except ImportError:
    from duckduckgo_search import DDGS   # older package name

from doc_reader import extract_text, read_document, DocumentError   # noqa: F401  (re-exported)
from database import (
    save_message, get_recent_memory,
    add_task, get_task, get_all_tasks, find_tasks_by_title,
    complete_task, delete_task, delete_all_tasks, delete_completed_tasks,
)

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    print("❌ GROQ_API_KEY not found! Check that .env sits next to app.py.", file=sys.stderr)
else:
    print("✅ Groq API key loaded")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Models get retired. Override without touching code:  JARVIS_MODEL=... in .env
MODEL = os.getenv("JARVIS_MODEL", "openai/gpt-oss-120b")
HISTORY_MESSAGES = 10
MAX_TOOL_STEPS = 6

# ─────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────

PERSONALITY = """You are JARVIS, a sharp, friendly and witty personal AI assistant, inspired by the AI from Iron Man.

Style:
- Speak like a knowledgeable friend, not a robot. Direct, warm, light humour when it fits.
- Keep answers concise unless the user asks for detail.
- If you can't do something, say so briefly and offer the nearest thing you can do.

Task rules (critical):
- You manage the user's tasks ONLY through your tools: add_task, list_tasks, complete_task, delete_task, clear_tasks.
- NEVER say a task was added, completed or deleted unless the tool result says "ok": true. If a tool fails, say so plainly.
- Task details in earlier messages may be outdated. For any question about the user's tasks, call list_tasks first.
- Never invent task ids or titles. If several tasks match, ask the user which one.
- Resolve relative dates (tomorrow, next Monday, 5 Sept) into YYYY-MM-DD using today's date. A bare weekday means the coming one. If the year is missing, use the next future occurrence.
- Do not call clear_tasks unless the user clearly asked to wipe tasks.

Knowledge:
- Use web_search for current events, live data, prices, scores or anything that may have changed recently. Don't search for things you know well.
"""


def build_system_prompt(document_context=None) -> str:
    today = datetime.date.today()
    prompt = PERSONALITY + f"\nToday is {today.strftime('%A, %B %d, %Y')} ({today.isoformat()}).\n"
    if document_context:
        prompt += f"""
The user attached a document. Excerpts retrieved for this question are below.
- Answer from these excerpts and be specific: use the document's own terms, numbers and names.
- If the excerpts don't contain the answer, say that instead of guessing.
- Do not search the web for questions about the document.

--- DOCUMENT EXCERPTS ---
{document_context}
--- END OF EXCERPTS ---
"""
    return prompt


# ─────────────────────────────────────────
# WEB SEARCH
# ─────────────────────────────────────────

def search_web(query: str, recent: bool = False) -> list:
    """DuckDuckGo search. Returns a list of {title, snippet, url}."""
    kwargs = {"region": "wt-wt", "max_results": 5}
    if recent:
        kwargs["timelimit"] = "m"      # last month only
    with DDGS() as ddgs:
        raw = list(ddgs.text(query, **kwargs))
    return [
        {"title": r.get("title", ""), "snippet": (r.get("body") or "")[:300], "url": r.get("href", "")}
        for r in raw
    ]


# ─────────────────────────────────────────
# TOOLS (what the LLM is allowed to do)
# ─────────────────────────────────────────

def _fn(name, description, properties, required=None):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required or []},
    }}


TASK_TOOLS = [
    _fn("add_task",
        "Create a new task. Call this whenever the user asks to add, schedule, note or be reminded of something to do.",
        {"title": {"type": "string", "description": "Short task title, e.g. 'Prepare cloud presentation'"},
         "due_date": {"type": "string", "description": "Due date as YYYY-MM-DD. Omit if the user gave none."},
         "description": {"type": "string", "description": "Optional extra details."}},
        ["title"]),
    _fn("list_tasks",
        "Get the user's current tasks from the database. Always call this before answering questions about their tasks.",
        {"status": {"type": "string", "enum": ["pending", "completed", "all"], "description": "Which tasks to return. Default pending."}}),
    _fn("complete_task",
        "Mark a task as completed. Identify it by task_id (preferred) or by part of its title.",
        {"task_id": {"type": "integer"}, "title_match": {"type": "string", "description": "Part of the task title"}}),
    _fn("delete_task",
        "Delete one task. Identify it by task_id (preferred) or by part of its title.",
        {"task_id": {"type": "integer"}, "title_match": {"type": "string", "description": "Part of the task title"}}),
    _fn("clear_tasks",
        "Delete MANY tasks at once. Only when the user explicitly asks to clear/remove all tasks or all completed tasks.",
        {"scope": {"type": "string", "enum": ["all", "completed"]}}, ["scope"]),
]

SEARCH_TOOL = _fn(
    "web_search",
    "Search the web for current or recent information.",
    {"query": {"type": "string"},
     "recent": {"type": "boolean", "description": "true to restrict to the last month (news, latest events)"}},
    ["query"])


def _task_view(t: dict) -> dict:
    return {"id": t["id"], "title": t["title"], "due_date": t["due_date"],
            "status": "completed" if t["completed"] else "pending"}


def _parse_due(value):
    """Returns (iso_date_or_None, error_or_None)."""
    if value is None:
        return None, None
    s = str(value).strip()
    if s.lower() in ("", "none", "null", "n/a"):
        return None, None
    try:
        return datetime.date.fromisoformat(s[:10]).isoformat(), None
    except ValueError:
        return None, f"due_date '{s}' is not a valid YYYY-MM-DD date"


def tool_add_task(title=None, due_date=None, description=None, **_):
    title = (title or "").strip()
    if not title:
        return {"ok": False, "error": "title is required"}
    due, err = _parse_due(due_date)
    if err:
        return {"ok": False, "error": err}

    # Don't create exact duplicates when the user repeats themselves
    for t in find_tasks_by_title(title):
        if t["title"].strip().lower() == title.lower() and (t["due_date"] or None) == due:
            return {"ok": True, "duplicate": True, "task": _task_view(t),
                    "note": "An identical pending task already exists; nothing new was added."}

    task_id = add_task(title, description, due)
    saved = get_task(task_id)          # read it back: proof that it is really in the database
    if not saved:
        return {"ok": False, "error": "task could not be saved"}
    return {"ok": True, "task": _task_view(saved)}


def tool_list_tasks(status="pending", **_):
    status = status if status in ("pending", "completed", "all") else "pending"
    tasks = get_all_tasks()
    if status == "pending":
        tasks = [t for t in tasks if not t["completed"]]
    elif status == "completed":
        tasks = [t for t in tasks if t["completed"]]
    return {"ok": True, "count": len(tasks), "tasks": [_task_view(t) for t in tasks[:50]]}


def _resolve_task(task_id, title_match, include_completed):
    """Find exactly one task. Returns (task, error_dict)."""
    if task_id not in (None, ""):
        try:
            tid = int(task_id)
        except (TypeError, ValueError):
            return None, {"ok": False, "error": "task_id must be an integer"}
        task = get_task(tid)
        if not task:
            return None, {"ok": False, "error": f"no task with id {tid}"}
        return task, None

    if title_match and str(title_match).strip():
        matches = find_tasks_by_title(str(title_match), include_completed)
        if not matches:
            return None, {"ok": False, "error": f"no task matches '{title_match}'"}
        if len(matches) > 1:
            exact = [m for m in matches if m["title"].strip().lower() == str(title_match).strip().lower()]
            if len(exact) == 1:
                return exact[0], None
            return None, {"ok": False, "error": "several tasks match; ask the user which one",
                          "candidates": [_task_view(m) for m in matches[:10]]}
        return matches[0], None

    return None, {"ok": False, "error": "provide task_id or title_match"}


def tool_complete_task(task_id=None, title_match=None, **_):
    task, err = _resolve_task(task_id, title_match, include_completed=False)
    if err:
        return err
    if task["completed"]:
        return {"ok": True, "already_completed": True, "task": _task_view(task)}
    complete_task(task["id"])
    return {"ok": True, "task": _task_view(get_task(task["id"]))}


def tool_delete_task(task_id=None, title_match=None, **_):
    task, err = _resolve_task(task_id, title_match, include_completed=True)
    if err:
        return err
    delete_task(task["id"])
    return {"ok": True, "deleted": _task_view(task)}


def tool_clear_tasks(scope=None, **_):
    if scope == "all":
        return {"ok": True, "deleted_count": delete_all_tasks(), "scope": "all"}
    if scope == "completed":
        return {"ok": True, "deleted_count": delete_completed_tasks(), "scope": "completed"}
    return {"ok": False, "error": "scope must be 'all' or 'completed'"}


def tool_web_search(query=None, recent=False, **_):
    if not query or not str(query).strip():
        return {"ok": False, "error": "query is required"}
    try:
        results = search_web(str(query).strip(), bool(recent))
    except Exception as e:
        return {"ok": False, "error": f"search failed: {e}"}
    return {"ok": True, "results": results} if results else {"ok": True, "results": [], "note": "no results"}


TOOL_FUNCS = {
    "add_task": tool_add_task,
    "list_tasks": tool_list_tasks,
    "complete_task": tool_complete_task,
    "delete_task": tool_delete_task,
    "clear_tasks": tool_clear_tasks,
    "web_search": tool_web_search,
}
MUTATING_TOOLS = {"add_task", "complete_task", "delete_task", "clear_tasks"}


def run_tool(name: str, raw_args) -> dict:
    func = TOOL_FUNCS.get(name)
    if func is None:
        return {"ok": False, "error": f"unknown tool '{name}'"}
    try:
        args = json.loads(raw_args) if raw_args else {}
        if not isinstance(args, dict):
            args = {}
    except json.JSONDecodeError:
        return {"ok": False, "error": "tool arguments were not valid JSON"}
    try:
        return func(**args)
    except TypeError as e:
        return {"ok": False, "error": f"bad arguments: {e}"}
    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "error": str(e)}


# ─────────────────────────────────────────
# LLM CALL (handles reasoning models + transient errors)
# ─────────────────────────────────────────

def _llm(messages, tools=None, effort="low", max_tokens=2048, temperature=0.5):
    """One chat-completion call.

    - max_completion_tokens is generous because reasoning models spend part of it on hidden thinking.
    - Retries once or twice on rate limits and on malformed tool calls (tool_use_failed).
    - Falls back to no reasoning_effort if a model/SDK rejects it.
    """
    kwargs = dict(model=MODEL, messages=messages, temperature=temperature, max_completion_tokens=max_tokens)
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    use_effort = effort
    for attempt in range(3):
        try:
            if use_effort:
                return client.chat.completions.create(reasoning_effort=use_effort, **kwargs)
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            text = str(e).lower()
            if use_effort and "reasoning" in text:
                use_effort = None
                continue
            if "tool_use_failed" in text and attempt < 2:
                continue
            if ("429" in text or "rate limit" in text or "rate_limit" in text) and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            raise
    raise RuntimeError("LLM call failed after retries")


# ─────────────────────────────────────────
# MAIN CHAT
# ─────────────────────────────────────────

_CLAIM_RE = re.compile(r"\b(added|created|saved|scheduled|completed|marked|deleted|removed|cleared)\b", re.I)
_GUARD_NUDGE = (
    "[system check] You have not called any task tool in this turn. If the user asked you to add, complete, "
    "delete or list tasks, call the correct tool now. Otherwise repeat your answer without implying that any "
    "task was changed."
)


def _claims_task_action(reply: str) -> bool:
    low = reply.lower()
    return "task" in low and bool(_CLAIM_RE.search(low))


def chat_with_meta(user_message: str, document_context: str = None) -> dict:
    """Run one conversational turn. Returns
    {"reply": str, "tasks_changed": bool, "tools_used": [names], "steps": int}"""
    try:
        history = [{"role": m["role"], "content": m["content"]} for m in get_recent_memory(HISTORY_MESSAGES)]
        messages = ([{"role": "system", "content": build_system_prompt(document_context)}]
                    + history + [{"role": "user", "content": user_message}])

        # While a document is attached, the model should answer from it, not browse the web
        tools = list(TASK_TOOLS) if document_context else TASK_TOOLS + [SEARCH_TOOL]
        effort = "medium" if document_context else "low"

        tools_used, tasks_changed, guard_used = [], False, False
        reply, steps = "", 0

        while steps < MAX_TOOL_STEPS:
            steps += 1
            resp = _llm(messages, tools, effort)
            msg = resp.choices[0].message
            calls = getattr(msg, "tool_calls", None) or []

            if calls:
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [{"id": c.id, "type": "function",
                                    "function": {"name": c.function.name,
                                                 "arguments": c.function.arguments or "{}"}} for c in calls],
                })
                for c in calls:
                    result = run_tool(c.function.name, c.function.arguments)
                    print(f"🛠️  {c.function.name}({c.function.arguments}) -> {str(result)[:160]}")
                    tools_used.append(c.function.name)
                    if c.function.name in MUTATING_TOOLS and result.get("ok") and not result.get("duplicate"):
                        tasks_changed = True
                    messages.append({"role": "tool", "tool_call_id": c.id,
                                     "content": json.dumps(result, ensure_ascii=False)})
                continue

            reply = (msg.content or "").strip()

            # Safety net: the model claims a task action but never called a tool
            if not tools_used and not guard_used and _claims_task_action(reply):
                guard_used = True
                print("⚠️  Reply claims a task action without a tool call - asking the model to verify")
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": _GUARD_NUDGE})
                continue
            break
        else:
            # Tool loop did not converge: force a plain answer
            resp = _llm(messages, None, "low")
            reply = (resp.choices[0].message.content or "").strip()

        if not reply:
            reply = ("Done." if tasks_changed else
                     "I didn't get a usable answer from the model that time. Please try again.")

        save_message("user", user_message)
        save_message("assistant", reply)
        return {"reply": reply, "tasks_changed": tasks_changed, "tools_used": tools_used, "steps": steps}

    except Exception as e:
        traceback.print_exc()
        return {"reply": f"JARVIS encountered an error: {e}", "tasks_changed": False,
                "tools_used": [], "steps": 0}


def chat(user_message: str, document_context: str = None) -> str:
    """Simple wrapper: just the reply text."""
    return chat_with_meta(user_message, document_context)["reply"]


# ─────────────────────────────────────────
# DAILY SUMMARY
# ─────────────────────────────────────────

def get_daily_summary() -> dict:
    """Task statistics for the dashboard."""
    all_tasks = get_all_tasks()
    pending = [t for t in all_tasks if not t["completed"]]
    completed = [t for t in all_tasks if t["completed"]]

    today = datetime.date.today()
    overdue, due_soon = [], []
    for task in pending:
        if task.get("due_date"):
            try:
                days_left = (datetime.date.fromisoformat(task["due_date"]) - today).days
            except ValueError:
                continue
            if days_left < 0:
                overdue.append(task)
            elif days_left <= 2:
                due_soon.append(task)

    return {
        "total": len(all_tasks),
        "pending": len(pending),
        "completed": len(completed),
        "overdue": len(overdue),
        "due_soon": len(due_soon),
        "pending_tasks": pending,
        "overdue_tasks": overdue,
        "due_soon_tasks": due_soon,
    }