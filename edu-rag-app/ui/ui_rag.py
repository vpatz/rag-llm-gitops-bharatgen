# ui_rag.py
import json
import requests
import sys
import gradio as gr
from pathlib import Path
import importlib
from typing import Generator, Tuple, List, Optional, Any

SETTINGS_PATH = Path(__file__).resolve().parent.parent / "src/settings.json"
with open(SETTINGS_PATH, 'r') as f:
    data = json.load(f)
port = int(data.get("backend_server_port"))
url = str(data.get("backend_url"))
rag_port = int(data.get("rag_ui_port"))

DEFAULT_BACKEND = f"{url}:{port}"

# ensure project root (parent dir that contains 'src') is on sys.path
_this_file = Path(__file__).resolve()
proj_root = None
cur = _this_file.parent
while True:
    if (cur / "src").is_dir():
        proj_root = cur
        break
    if cur == cur.parent:
        break
    cur = cur.parent
if proj_root is None:
    proj_root = _this_file.resolve().parents[1]

if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

retrieval_module = importlib.import_module("src.retrieve.retrieval_utils")
format_chunks = getattr(retrieval_module, "show_document_content", None)

# ---- Helpers ----

def call_reference(base_url: str, prompt: str,
                   num_chunks_post_rrf: int = 10,
                   num_docs_reranker: int = 3,
                   use_reranker: bool = True,
                   language: Optional[str] = None,
                   subject: Optional[str] = None,
                   class_level: Optional[str] = None,
                   timeout: float = 10.0) -> Tuple[bool, List[dict], str]:
    url = base_url.rstrip("/") + "/reference"
    payload = {
        "prompt": prompt,
        "num_chunks_post_rrf": num_chunks_post_rrf,
        "num_docs_reranker": num_docs_reranker,
        "use_reranker": use_reranker,
        "language": language,
        "subject": subject,
        "class_level": class_level,
    }
    payload = {k: v for k, v in payload.items() if v not in (None, "", [])}
    try:
        r = requests.post(url, json=payload, timeout=timeout)
    except requests.RequestException as e:
        return False, [], f"Request error calling /reference: {e}"
    if r.status_code == 404:
        return False, [], f"{url} returned 404"
    if not r.ok:
        # try to parse body for error message
        try:
            body = r.json()
            return False, [], f"/reference error {r.status_code}: {body}"
        except Exception:
            return False, [], f"/reference error {r.status_code}: {r.text[:200]}"
    try:
        j = r.json()
        docs = j.get("documents", [])
        docs = [d if isinstance(d, dict) else {"page_content": str(d)} for d in docs]
        return True, docs, ""
    except Exception as e:
        return False, [], [], f"Failed to parse /reference JSON: {e}"

def parse_sse_line(line: bytes) -> Optional[str]:
    """
    Given a raw SSE line (bytes), returns the payload string (after 'data: ')
    or None if not a data line / empty.
    """
    if not line:
        return None
    try:
        s = line.decode("utf-8")
    except Exception:
        return None
    s = s.strip()
    if not s:
        return None
    if s.startswith("data:"):
        return s[len("data:"):].strip()
    return None

def call_chat_stream(base_url: str, prompt: str, docs: List[dict],
                     max_tokens: int = 512, temperature: float = 0.0,
                     stop_words = None, stream: bool = True,
                     num_chunks_post_rrf: int = 10,
                     num_docs_reranker: int = 3,
                     use_reranker: bool = True,
                     timeout: float = 60.0) -> requests.Response:
    """
    Calls /v1/chat/completions and includes retrieved_documents in the payload so the backend
    uses them instead of re-running retrieval.
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "retrieved_documents": docs,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop_words,
        "stream": stream,
        "num_chunks_post_rrf": num_chunks_post_rrf,
        "num_docs_reranker": num_docs_reranker,
        "use_reranker": use_reranker
    }
    # Remove None values to keep payload clean
    payload = {k: v for k, v in payload.items() if v is not None}
    # Use stream=True for streaming responses
    resp = requests.post(url, json=payload, stream=stream, timeout=timeout)
    return resp

# ---- Gradio generator which yields updated answer + docs_html as it streams ----

def ask_and_stream(backend_url: str, query: str,
                   use_reranker: bool, num_chunks_post_rrf: int,
                   num_docs_reranker: int, max_tokens: int,
                   temperature: float,
                   language: Optional[str],
                   subject: Optional[str],
                   class_level: Optional[str],
                   stream: bool = True) -> Generator[Tuple[str, str], None, None]:

    backend = backend_url.strip() or DEFAULT_BACKEND
    q = (query or "").strip()
    if not q:
        yield "Please enter a non-empty query.", "<div>No docs.</div>"
        return

    ok, docs, err = call_reference(
        backend, q,
        num_chunks_post_rrf=num_chunks_post_rrf,
        num_docs_reranker=num_docs_reranker,
        use_reranker=use_reranker,
        language=language,
        subject=subject,
        class_level=class_level
    )

    if not ok:
        docs_html = f"<div style='color: red'>Error obtaining reference docs: {err}</div>"
        yield "", docs_html
        return

    # Create docs HTML using the provided renderer
    scores = [1.] * len(docs)
    if callable(format_chunks):
        docs_html = format_chunks(docs, scores) if docs else "<div>No reference docs returned.</div>"
    else:
        docs_html = "<div>No reference docs returned.</div>"

    # Yield initial state (docs visible immediately)
    yield "Requesting answer...", docs_html

    # Call chat endpoint and pass retrieved documents so backend will reuse them
    try:
        resp = call_chat_stream(backend, q, docs,
                                max_tokens=max_tokens, temperature=temperature,
                                stream=stream,
                                num_chunks_post_rrf=num_chunks_post_rrf,
                                num_docs_reranker=num_docs_reranker,
                                use_reranker=use_reranker)
    except requests.RequestException as e:
        yield f"Request error calling chat endpoint: {e}", docs_html
        return

    if resp.status_code >= 400:
        # try to extract JSON error
        try:
            body = resp.json()
            yield f"Chat endpoint error {resp.status_code}: {body}", docs_html
        except Exception:
            yield f"Chat endpoint error {resp.status_code}: {resp.text[:200]}", docs_html
        return

    # Streaming SSE handling
    accumulated = ""
    content_type = resp.headers.get("Content-Type", "")
    if "text/event-stream" in content_type or stream:
        try:
            for raw_line in resp.iter_lines(decode_unicode=False):
                payload = parse_sse_line(raw_line)
                if payload is None:
                    continue
                try:
                    parsed = json.loads(payload)
                    chunk_text = ""
                    if isinstance(parsed, dict):
                        # common vllm/v0 shapes: {"choices":[{"delta":{"content":"..."}}]}
                        choices = parsed.get("choices")
                        if choices and isinstance(choices, list) and choices:
                            first = choices[0]
                            if isinstance(first, dict):
                                delta = first.get("delta")
                                if isinstance(delta, dict) and "content" in delta:
                                    chunk_text = delta.get("content") or ""
                                elif "text" in first:
                                    chunk_text = first.get("text") or ""
                        elif "text" in parsed:
                            chunk_text = parsed.get("text") or ""
                        else:
                            # fallback: find first string value
                            for v in parsed.values():
                                if isinstance(v, str) and v.strip():
                                    chunk_text = v
                                    break
                    else:
                        chunk_text = str(parsed)
                    if chunk_text:
                        accumulated += chunk_text
                        yield accumulated, docs_html
                except json.JSONDecodeError:
                    # non-json payload
                    txt = payload.strip()
                    if txt:
                        accumulated += txt
                        yield accumulated, docs_html
            # finished
            yield accumulated or "[no content received]", docs_html
            return
        except requests.RequestException as e:
            yield f"Streaming request error: {e}", docs_html
            return
    else:
        # Non-streaming JSON
        try:
            j = resp.json()
            answer = ""
            if isinstance(j, dict):
                choices = j.get("choices")
                if choices and isinstance(choices, list) and len(choices) > 0:
                    c0 = choices[0]
                    if isinstance(c0, dict):
                        msg = c0.get("message") or c0.get("delta") or c0
                        if isinstance(msg, dict) and "content" in msg:
                            answer = msg.get("content") or ""
                        elif "text" in c0:
                            answer = c0.get("text") or ""
                if not answer:
                    for k in ("answer", "result", "response", "text"):
                        if k in j:
                            answer = str(j[k])
                            break
            if not answer:
                answer = json.dumps(j)
            yield answer, docs_html
            return
        except Exception as e:
            yield f"Failed to parse non-streaming chat response: {e}", docs_html
            return

# -------------------------
# Gradio UI
# -------------------------
def build_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# RAG")
        with gr.Row():
            backend_in = gr.Textbox(label="Backend base URL", value=DEFAULT_BACKEND, lines=1)
            check_health = gr.Button("Check Health / DB Status")
        with gr.Row():
            with gr.Column(scale=2):
                q_in = gr.Textbox(label="Question", placeholder="Write your question here...", lines=4)

            with gr.Column(scale=1):
                gr.Markdown("### Filters")
                language_in = gr.Dropdown(["", "English", "Hindi"], label="Language", value="")
                subject_in = gr.Dropdown(["", "Biology", "Chemistry", "Maths", "Physics", "Science"], label="Subject", value="")
                class_in = gr.Dropdown(["", "Class-10", "Class-11", "Class-12"], label="Class Level", value="")

            with gr.Column(scale=1):
                gr.Markdown("### Generation")
                use_reranker = gr.Checkbox(label="Use reranker", value=True)
                num_chunks_post_rrf = gr.Slider(label="num_chunks_post_rrf", minimum=1, maximum=50, step=1, value=10)
                num_docs_reranker = gr.Slider(label="num_docs_reranker", minimum=1, maximum=20, step=1, value=3)
                max_tokens = gr.Slider(label="max_tokens", minimum=16, maximum=2048, step=1, value=512)
                temperature = gr.Slider(label="temperature", minimum=0.0, maximum=1.0, step=0.01, value=0.0)

        ask_btn = gr.Button("Ask")
        with gr.Row():
            answer_out = gr.Textbox(label="Answer (streaming)", lines=10, interactive=False)
            docs_out = gr.HTML("<div>No docs yet.</div>")

        def check_status(backend_url: str) -> str:
            b = backend_url.strip() or DEFAULT_BACKEND
            try:
                r = requests.get(b.rstrip("/") + "/health", timeout=4)
                if r.ok:
                    try:
                        r2 = requests.get(b.rstrip("/") + "/db-status", timeout=4)
                        if r2.ok:
                            js = r2.json()
                            return f"Health OK. DB status: {js}"
                        else:
                            return f"Health OK. /db-status returned {r2.status_code}: {r2.text}"
                    except Exception:
                        return "Health OK. Failed to query /db-status."
                else:
                    return f"/health returned {r.status_code}: {r.text[:200]}"
            except Exception as e:
                return f"Health check failed: {e}"

        check_health.click(fn=check_status, inputs=[backend_in], outputs=[docs_out])

        ask_btn.click(
            fn=ask_and_stream,
            inputs=[
                backend_in,
                q_in,
                use_reranker,
                num_chunks_post_rrf,
                num_docs_reranker,
                max_tokens,
                temperature,
                language_in,
                subject_in,
                class_in
            ],
            outputs=[answer_out, docs_out]
        )

    return demo

if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=rag_port, share=False)
