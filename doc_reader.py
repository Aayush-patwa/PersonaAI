# doc_reader.py - Extract text from uploaded documents (PDF, DOCX, TXT)

try:
    import pymupdf as fitz      # newer PyMuPDF name
except ImportError:             # older installs
    import fitz


class DocumentError(Exception):
    """Raised when a document can't be turned into usable text.
    The message is written for the end user and is safe to show in the UI."""


SUPPORTED_EXTENSIONS = {"pdf", "docx", "txt"}


def _read_txt(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _read_docx(filepath: str) -> str:
    from docx import Document
    doc = Document(filepath)
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(c.text.strip() for c in row.cells if c.text.strip())
            if row_text:
                parts.append(row_text)
    return "\n".join(parts)


def _read_pdf(filepath: str) -> str:
    doc = fitz.open(filepath)
    try:
        if doc.needs_pass:
            raise DocumentError("This PDF is password-protected. Remove the password and upload it again.")
        text = "\n".join(page.get_text() for page in doc)
        if not text.strip():
            n_images = sum(len(page.get_images()) for page in doc)
            if n_images > 0:
                raise DocumentError(
                    f"This looks like a scanned PDF ({n_images} image(s), no text layer), so there is "
                    "nothing to read. Run OCR on it first, or upload a text-based PDF / .docx / .txt."
                )
            raise DocumentError("This PDF has no extractable text.")
        return text
    finally:
        doc.close()


def extract_text(filepath: str) -> str:
    """Return the FULL text of a document (no truncation).

    Raises DocumentError with a user-friendly message if the file can't be read.
    """
    ext = filepath.lower().rsplit(".", 1)[-1]
    if ext not in SUPPORTED_EXTENSIONS:
        raise DocumentError(f"Unsupported file type: .{ext}. Please upload PDF, DOCX or TXT files.")

    try:
        if ext == "pdf":
            text = _read_pdf(filepath)
        elif ext == "docx":
            text = _read_docx(filepath)
        else:
            text = _read_txt(filepath)
    except DocumentError:
        raise
    except Exception as e:
        raise DocumentError(f"Could not open the file: {e}")

    text = text.replace("\x00", "")
    if not text.strip():
        raise DocumentError("This document appears to be empty.")
    return text


def read_document(filepath: str, max_chars: int = 6000) -> str:
    """Backward-compatible helper: truncated text, or the error message as a string.
    Use extract_text() for indexing so the whole document is covered."""
    try:
        text = extract_text(filepath)
    except DocumentError as e:
        return str(e)
    return text[:max_chars] + "..." if len(text) > max_chars else text