# database.py - Handles all data storage for JARVIS
import sqlite3
import datetime
import os

# Absolute paths: the app finds its database no matter which folder you launch it from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "database")
DB_PATH = os.path.join(DB_DIR, "jarvis.db")
os.makedirs(DB_DIR, exist_ok=True)


def get_connection():
    """Get a database connection (rows behave like dicts)."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _query(sql, params=()):
    """Run a SELECT and return a list of dicts."""
    conn = get_connection()
    try:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def _execute(sql, params=()):
    """Run a write statement. Returns (lastrowid, rowcount)."""
    conn = get_connection()
    try:
        cur = conn.execute(sql, params)
        conn.commit()
        return cur.lastrowid, cur.rowcount
    finally:
        conn.close()


def initialize_database():
    """Create all tables if they don't exist."""
    conn = get_connection()
    try:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                due_date TEXT,
                completed INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                completed_at TEXT
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                original_name TEXT NOT NULL,
                file_type TEXT,
                file_size INTEGER,
                summary TEXT,
                uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
    finally:
        conn.close()
    print("✅ Database initialized successfully")


# ─────────────────────────────────────────
# TASK FUNCTIONS
# ─────────────────────────────────────────

def add_task(title, description=None, due_date=None):
    """Add a new task. Returns the new task id."""
    description = (description or "").strip() or None
    if isinstance(due_date, str):
        due_date = due_date.strip() or None
    task_id, _ = _execute(
        'INSERT INTO tasks (title, description, due_date, created_at) VALUES (?, ?, ?, ?)',
        (title, description, due_date, datetime.datetime.now().isoformat())
    )
    return task_id


def get_task(task_id):
    """Get one task by id (or None)."""
    rows = _query('SELECT * FROM tasks WHERE id = ?', (task_id,))
    return rows[0] if rows else None


def get_all_tasks():
    """All tasks: pending first (soonest due date first), then completed."""
    return _query(
        'SELECT * FROM tasks '
        'ORDER BY completed ASC, due_date IS NULL, due_date ASC, id DESC'
    )


def get_pending_tasks():
    """All incomplete tasks, soonest due date first (undated last)."""
    return _query(
        'SELECT * FROM tasks WHERE completed = 0 '
        'ORDER BY due_date IS NULL, due_date ASC, id DESC'
    )


def find_tasks_by_title(fragment, include_completed=False):
    """Case-insensitive title search, used when the user names a task instead of an id."""
    fragment = (fragment or "").strip()
    if not fragment:
        return []
    escaped = fragment.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
    sql = "SELECT * FROM tasks WHERE title LIKE ? ESCAPE '\\'"
    if not include_completed:
        sql += " AND completed = 0"
    sql += " ORDER BY id DESC"
    return _query(sql, (f"%{escaped}%",))


def complete_task(task_id):
    """Mark a task as complete."""
    _, affected = _execute(
        'UPDATE tasks SET completed = 1, completed_at = ? WHERE id = ?',
        (datetime.datetime.now().isoformat(), task_id)
    )
    return affected > 0


def delete_task(task_id):
    """Delete a task."""
    _, affected = _execute('DELETE FROM tasks WHERE id = ?', (task_id,))
    return affected > 0


def delete_all_tasks():
    """Delete every task. Returns how many were removed."""
    _, affected = _execute('DELETE FROM tasks')
    return affected


def delete_completed_tasks():
    """Delete only completed tasks. Returns how many were removed."""
    _, affected = _execute('DELETE FROM tasks WHERE completed = 1')
    return affected


def get_due_reminders():
    """Incomplete tasks that are overdue, due today or due tomorrow."""
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    return _query(
        '''SELECT * FROM tasks
           WHERE completed = 0
           AND due_date IS NOT NULL
           AND date(due_date) <= ?
           ORDER BY due_date ASC''',
        (tomorrow.isoformat(),)
    )


# ─────────────────────────────────────────
# MEMORY FUNCTIONS
# ─────────────────────────────────────────

def save_message(role, content):
    """Save a chat message to memory."""
    _execute(
        'INSERT INTO memory (role, content, timestamp) VALUES (?, ?, ?)',
        (role, content, datetime.datetime.now().isoformat())
    )


def get_recent_memory(limit=10):
    """Recent conversation history, oldest first."""
    rows = _query('SELECT role, content FROM memory ORDER BY id DESC LIMIT ?', (limit,))
    return list(reversed(rows))


def clear_memory():
    """Clear all chat memory."""
    _execute('DELETE FROM memory')


# ─────────────────────────────────────────
# DOCUMENT FUNCTIONS
# ─────────────────────────────────────────

def save_document(filename, original_name, file_type, file_size, summary=None):
    """Save document metadata. Returns the new document id."""
    doc_id, _ = _execute(
        '''INSERT INTO documents (filename, original_name, file_type, file_size, summary, uploaded_at)
           VALUES (?, ?, ?, ?, ?, ?)''',
        (filename, original_name, file_type, file_size, summary, datetime.datetime.now().isoformat())
    )
    return doc_id


def get_document(doc_id):
    """Get one document record by id (or None)."""
    rows = _query('SELECT * FROM documents WHERE id = ?', (doc_id,))
    return rows[0] if rows else None


def get_all_documents():
    """Get all uploaded documents, newest first."""
    return _query('SELECT * FROM documents ORDER BY id DESC')


def delete_document(doc_id):
    """Delete a document record."""
    _, affected = _execute('DELETE FROM documents WHERE id = ?', (doc_id,))
    return affected > 0


# Initialize database when this file is imported
initialize_database()