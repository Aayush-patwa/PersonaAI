# app.py - Flask web server for JARVIS
import os
import time
import uuid
import datetime
import threading

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

load_dotenv()

from agent import chat_with_meta, get_daily_summary, MODEL
from doc_reader import extract_text, DocumentError, SUPPORTED_EXTENSIONS
from database import (
    initialize_database,
    add_task, get_all_tasks,
    complete_task, delete_task,
    save_document, get_document, get_all_documents, delete_document,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DEBUG = True

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB max upload


def _rag():
    """Import rag lazily: it pulls in torch, and the Flask reloader starts two processes."""
    import rag
    return rag


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in SUPPORTED_EXTENSIONS


def doc_key(doc: dict) -> str:
    """Index key = the document's unique stored filename (stays correct even if the DB is reset)."""
    return os.path.splitext(doc["filename"])[0]


def valid_date(value):
    """Return (iso_date_or_None, ok)."""
    if not value:
        return None, True
    try:
        return datetime.date.fromisoformat(str(value)[:10]).isoformat(), True
    except ValueError:
        return None, False


@app.after_request
def no_cache_for_api(resp):
    # Task lists must never come from a browser cache
    if request.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"success": False, "message": "File too large (max 16 MB)."}), 413


# ─────────────────────────────────────────
# MAIN ROUTE
# ─────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


# ─────────────────────────────────────────
# CHAT API
# ─────────────────────────────────────────

def build_document_context(doc_id: int, message: str):
    """RAG step: return (excerpts_or_None, error_message_or_None)."""
    doc = get_document(doc_id)
    if not doc:
        return None, "That document no longer exists. Pick another one from the dropdown."

    rag = _rag()
    key = doc_key(doc)
    if not rag.index_is_current(key):
        print("📄 Building RAG index on the fly...")
        text = extract_text(os.path.join(UPLOAD_FOLDER, doc["filename"]))
        rag.build_index(key, text)

    context = rag.retrieve(key, message)
    print(f"🔍 RAG retrieved {len(context) if context else 0} chars for: {message[:60]}")
    return context, None


@app.route("/api/chat", methods=["POST"])
def chat_api():
    t0 = time.perf_counter()
    try:
        data = request.get_json(silent=True) or {}
        message = (data.get("message") or "").strip()
        doc_id = data.get("doc_id")

        if not message:
            return jsonify({"success": False, "response": "Please type a message."})

        document_context = None
        if doc_id:
            try:
                document_context, problem = build_document_context(int(doc_id), message)
            except DocumentError as e:
                problem = f"I couldn't read that document: {e}"
            if problem:
                return jsonify({"success": True, "response": problem, "tasks_changed": False})

        result = chat_with_meta(message, document_context)
        return jsonify({
            "success": True,
            "response": result["reply"],
            "tasks_changed": result["tasks_changed"],
            "tools_used": result["tools_used"],
            "timestamp": datetime.datetime.now().strftime("%H:%M"),
            "latency_ms": int((time.perf_counter() - t0) * 1000),
        })
    except Exception as e:
        return jsonify({"success": False, "response": f"Error: {e}"})


# ─────────────────────────────────────────
# TASK APIs
# ─────────────────────────────────────────

@app.route("/api/tasks", methods=["GET"])
def get_tasks():
    try:
        return jsonify({"success": True, "tasks": get_all_tasks()})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/tasks/add", methods=["POST"])
def add_task_api():
    try:
        data = request.get_json(silent=True) or {}
        title = (data.get("title") or "").strip()
        description = (data.get("description") or "").strip()
        due_date, ok = valid_date(data.get("due_date"))

        if not title:
            return jsonify({"success": False, "message": "Task title cannot be empty."})
        if not ok:
            return jsonify({"success": False, "message": "Due date must look like YYYY-MM-DD."})

        task_id = add_task(title, description, due_date)
        return jsonify({"success": True, "message": "Task added successfully!", "task_id": task_id})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@app.route("/api/tasks/<int:task_id>/complete", methods=["POST"])
def complete_task_api(task_id):
    try:
        if complete_task(task_id):
            return jsonify({"success": True, "message": "Task marked as complete!"})
        return jsonify({"success": False, "message": "Task not found."})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@app.route("/api/tasks/<int:task_id>/delete", methods=["DELETE"])
def delete_task_api(task_id):
    try:
        if delete_task(task_id):
            return jsonify({"success": True, "message": "Task deleted!"})
        return jsonify({"success": False, "message": "Task not found."})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@app.route("/api/summary", methods=["GET"])
def summary_api():
    try:
        return jsonify({"success": True, "summary": get_daily_summary()})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# ─────────────────────────────────────────
# DOCUMENT APIs
# ─────────────────────────────────────────

@app.route("/api/documents", methods=["GET"])
def get_documents():
    try:
        return jsonify({"success": True, "documents": get_all_documents()})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/documents/upload", methods=["POST"])
def upload_document():
    filepath = None
    try:
        file = request.files.get("file")
        if file is None:
            return jsonify({"success": False, "message": "No file provided."})
        if file.filename == "":
            return jsonify({"success": False, "message": "No file selected."})
        if not allowed_file(file.filename):
            return jsonify({"success": False, "message": "Only PDF, DOCX and TXT files are allowed."})

        ext = file.filename.rsplit(".", 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(filepath)
        file_size = os.path.getsize(filepath)

        # 1) Read the WHOLE document (scanned / empty / protected files are rejected here)
        try:
            text = extract_text(filepath)
        except DocumentError as e:
            os.remove(filepath)
            return jsonify({"success": False, "message": str(e)})

        # 2) Build the RAG index before saving the record, so a failure leaves nothing half-done
        key = os.path.splitext(unique_filename)[0]
        try:
            n_chunks = _rag().build_index(key, text)
        except Exception as e:
            os.remove(filepath)
            return jsonify({"success": False, "message": f"Could not index the document: {e}"})

        # 3) Save metadata
        flat = " ".join(text.split())
        summary = flat[:200] + ("..." if len(flat) > 200 else "")
        doc_id = save_document(
            filename=unique_filename,
            original_name=secure_filename(file.filename) or f"document.{ext}",
            file_type=ext,
            file_size=file_size,
            summary=summary,
        )
        return jsonify({
            "success": True,
            "message": "Document uploaded and indexed!",
            "doc_id": doc_id,
            "original_name": file.filename,
            "chunks": n_chunks,
        })
    except Exception as e:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"success": False, "message": str(e)})


@app.route("/api/documents/<int:doc_id>/delete", methods=["DELETE"])
def delete_document_api(doc_id):
    try:
        doc = get_document(doc_id)
        if not doc:
            return jsonify({"success": False, "message": "Document not found."})

        filepath = os.path.join(UPLOAD_FOLDER, doc["filename"])
        if os.path.exists(filepath):
            os.remove(filepath)
        _rag().delete_index(doc_key(doc))
        delete_document(doc_id)
        return jsonify({"success": True, "message": "Document deleted!"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


# ─────────────────────────────────────────
# STATUS + SETTINGS APIs
# ─────────────────────────────────────────

@app.route("/api/status", methods=["GET"])
def status():
    now = datetime.datetime.now()
    return jsonify({
        "status": "online",
        "agent": "JARVIS",
        "model": MODEL,
        "time": now.strftime("%H:%M"),
        "date": now.strftime("%A, %B %d, %Y"),
    })


@app.route("/api/settings/test-email", methods=["POST"])
def test_email():
    try:
        from reminder import send_test_email
        if send_test_email():
            return jsonify({"success": True, "message": "Test email sent!"})
        return jsonify({"success": False, "message": "Failed. Check your Gmail credentials in .env"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@app.route("/api/settings/check-reminders", methods=["POST"])
def check_reminders_api():
    try:
        from reminder import check_and_send_reminders
        check_and_send_reminders()
        return jsonify({"success": True, "message": "Reminder check complete!"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@app.route("/api/settings/clear-memory", methods=["POST"])
def clear_memory_api():
    try:
        from database import clear_memory
        clear_memory()
        return jsonify({"success": True, "message": "Memory cleared!"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


# ─────────────────────────────────────────
# BACKGROUND JOBS
# ─────────────────────────────────────────

def start_scheduler():
    """Email reminders every day at 09:00 (only while this app is running)."""
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from reminder import check_and_send_reminders
        scheduler = BackgroundScheduler()
        scheduler.add_job(check_and_send_reminders, "cron", hour=9, minute=0,
                          id="daily_reminders", replace_existing=True, misfire_grace_time=3600)
        scheduler.start()
        print("⏰ Reminder scheduler started (daily at 09:00)")
    except Exception as e:
        print(f"⚠️ Scheduler not started: {e}")


def is_serving_process():
    # With Flask's reloader, only the child process (WERKZEUG_RUN_MAIN=true) serves requests.
    return os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not DEBUG


if __name__ == "__main__":
    initialize_database()
    print("=" * 60)
    print("🤖 JARVIS - Personal AI Assistant")
    print(f"🧠 Model: {MODEL}")
    print("=" * 60)
    print("🌐 Open: http://localhost:5000")
    print("=" * 60)

    if is_serving_process():
        threading.Thread(target=lambda: _rag().warmup(), daemon=True).start()
        start_scheduler()

    app.run(debug=DEBUG, port=5000)