# PersonaAI: RAG-Powered Personal AI Assistant

PersonaAI is a locally run AI assistant that answers questions about your own documents and manages your tasks through natural language. It combines a hybrid retrieval-augmented generation (RAG) pipeline with a tool-calling agent, so every task action in the chat is a real database operation that the web interface reflects immediately. In the web interface the assistant appears as "JARVIS".

## Features

- **Document question answering.** Upload PDF, DOCX or TXT files and ask questions about them. The full document is indexed, not just the first pages.
- **Hybrid retrieval.** Dense search (sentence-transformers + FAISS) and keyword search (BM25) are combined with Reciprocal Rank Fusion.
- **Task management by chat or UI.** Add, list, complete and delete tasks by typing naturally ("remind me to submit the report next Friday") or through the Tasks tab. Both use the same SQLite database.
- **Verified actions.** The language model manages tasks only through function calls. It reports an action as done only after the database confirms it, and a guard catches replies that claim an action without calling a tool.
- **Web search.** Current-events questions are answered with live DuckDuckGo results, chosen by the model as a tool.
- **Conversation memory.** Recent messages are stored in SQLite and passed back as context.
- **Email reminders (optional).** A daily job at 09:00 emails tasks that are overdue, due today or due tomorrow. It runs only while the app is running and needs a Gmail app password.
- **Evaluation harness.** A script that measures retrieval quality, answer quality and latency, and runs ablations over retrieval settings.

## Architecture

```
Browser (HTML / CSS / JavaScript)
        |  JSON over REST (fetch)
        v
Flask API  (app.py)
   |-- agent.py     tool-calling loop on Groq
   |                tools: add_task, list_tasks, complete_task,
   |                       delete_task, clear_tasks, web_search
   |-- rag.py       chunk -> embed -> FAISS + BM25 -> RRF fusion
   |-- doc_reader.py  PDF / DOCX / TXT text extraction
   |-- database.py  SQLite: tasks, memory, documents
   |-- reminder.py  Gmail SMTP reminders (APScheduler, daily 09:00)
```

### How a chat turn works

1. The app loads the last 10 messages and builds a system prompt containing today's date and the task rules.
2. If a document is selected, the question is used to retrieve the 4 most relevant excerpts, which are added to the prompt.
3. The model either answers directly or calls a tool. Tool results (including the saved database row) are passed back to the model.
4. The final reply is returned to the browser together with a flag saying whether tasks changed, so the Tasks tab can refresh.

### How retrieval works

- Documents are split into 160-word chunks with 40 words of overlap. The chunk size stays below the 256-token input limit of `all-MiniLM-L6-v2`.
- Each chunk is embedded and stored in a FAISS index (cosine similarity). A BM25 index is built from the same chunks.
- At query time, the top candidates from both retrievers are merged with Reciprocal Rank Fusion and the top 4 chunks go to the model.
- Indexes are stored per document under `database/vector_store/` and rebuilt automatically if the chunking settings change.

## Evaluation

The pipeline was evaluated on 30 questions across 3 documents (a data processing agreement, a NLP project report and an ML system design document). Questions were reworded in casual, natural phrasing. Each question has a reference answer and a short verbatim evidence phrase from the source.

| Metric | Result |
|---|---|
| Hit Rate@4 (evidence in the 4 retrieved chunks) | 93.3% (28 of 30) |
| MRR | 0.739 |
| Answer correctness (LLM-as-judge) | 93.3% |
| Faithfulness (LLM-as-judge) | 90.0% |
| Retrieval latency | 41 ms p50 / 219 ms p95 |
| End-to-end latency (retrieval + answer) | 5.0 s p50 / 6.9 s p95 |

### Limitations of this evaluation

- Small sample: 30 questions, so one question is 3.3 percentage points. Differences between configurations of a few points may be noise.
- Three documents only, and the questions were drafted from those documents.
- The same model generates and judges answers, which can bias the scores. The judge was occasionally too lenient or too strict.
- Two questions missed retrieval (vocabulary mismatch with the source, and a fact repeated elsewhere) and one failed at generation despite correct retrieval. Per-question details are in `eval_results.json`.
- There are no questions whose answer is absent from the documents and none that span several documents.

### Reproduce

```
python eval_rag.py --ablate            # compare modes and chunk sizes, no LLM calls
python eval_rag.py --retrieval-only    # retrieval metrics only
python eval_rag.py                     # full evaluation with LLM answers and judge
```

The test documents are not included in this repository. Upload your own documents in the app, then write `eval_set.json` as a list of entries with `document` (file name), `question`, `reference` (correct answer) and `evidence` (a short phrase copied word for word from the source). The script rejects entries whose evidence phrase is not found in the document.

## Tech stack

| Area | Tools |
|---|---|
| Backend | Python, Flask |
| LLM | Groq API, `openai/gpt-oss-120b` (function calling; configurable) |
| Embeddings and search | sentence-transformers (`all-MiniLM-L6-v2`), FAISS, BM25 (own implementation), Reciprocal Rank Fusion |
| Storage | SQLite |
| Documents | PyMuPDF, python-docx |
| Web search | DuckDuckGo (`ddgs`) |
| Scheduling and email | APScheduler, Gmail SMTP |
| Frontend | HTML, CSS, JavaScript |

## Setup

Requires Python 3.10 or newer and a free Groq API key from [console.groq.com](https://console.groq.com).

```
git clone https://github.com/Aayush-patwa/PersonaAI.git
cd PersonaAI

python -m venv venv
venv\Scripts\activate          # macOS / Linux: source venv/bin/activate

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your values:

```
GROQ_API_KEY=your_groq_key_here
JARVIS_MODEL=openai/gpt-oss-120b
GMAIL_ADDRESS=
GMAIL_APP_PASSWORD=
```

Only `GROQ_API_KEY` is required. The Gmail values enable email reminders and need a Google app password, not your normal password. If Groq retires the default model, set `JARVIS_MODEL` to any tool-calling model available on your account.

Run the app:

```
python app.py
```

Open `http://localhost:5000`. The first start downloads the embedding model (about 90 MB).

## Known limitations

- Scanned PDFs are rejected because there is no OCR step.
- Single-user and local only: there is no authentication and the app uses Flask's development server.
- Email reminders run only while the app is running.
- Retrieval uses only the latest message, so short follow-ups such as "explain more" retrieve poorly.

## Roadmap

- Query rewriting using chat history and a cross-encoder re-ranker
- Questions across multiple documents
- A persistent user profile that the assistant updates over time
- OCR for scanned PDFs
- Docker deployment and voice input

## License

MIT