# ui_rag.py
import json
import requests
import base64
import gradio as gr
from pathlib import Path
from typing import Generator, Tuple, List, Optional, Any

SETTINGS_PATH = Path(__file__).resolve().parent.parent / "src/settings.json"
with open(SETTINGS_PATH, 'r') as f:
    data = json.load(f)
port = int(data.get("backend_server_port"))
url = str(data.get("backend_url"))

DEFAULT_BACKEND = f"{url}:{port}"

# ---- Helpers ----

def call_reference(base_url: str, prompt: str,
                   num_chunks_post_rrf: int = 10,
                   num_docs_reranker: int = 3,
                   use_reranker: bool = True,
                   timeout: float = 10.0) -> Tuple[bool, List[dict], str]:
    url = base_url.rstrip("/") + "/reference"
    payload = {
        "prompt": prompt,
        "num_chunks_post_rrf": num_chunks_post_rrf,
        "num_docs_reranker": num_docs_reranker,
        "use_reranker": use_reranker
    }
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

# -------------------------
# Document formatting helpers
# -------------------------
def format_table_html(table_html: str) -> str:
    """
    Minimal table formatting wrapper. Replace with your own richer formatter if needed.
    """
    # If the backend already returns valid HTML for the table, just return it.
    # Otherwise wrap plain CSV/TSV/markdown in a <pre> block for readability.
    if table_html.strip().lower().startswith("<table"):
        return table_html
    # simple escape + <pre>
    safe = table_html.replace("<", "&lt;").replace(">", "&gt;")
    return f"<pre style='white-space: pre-wrap; font-family: monospace;'>{safe}</pre>"

def show_document_content(retrieved_documents: List[dict], scores: List[float]) -> str:
    """
    Renders retrieved documents into HTML. Uses the user's provided format.
    """
    html_content = ""
    for idx, (doc, score) in enumerate(zip(retrieved_documents, scores)):
        # doc is expected to be a dict with fields like 'type', 'filename', 'source', 'page_content', 'chunk_id'
        doc_type = doc.get("type", "text")
        # document_header = f'<h4>Document {idx + 1} (Score: {score:.4f}), (Doc: {doc.get("filename")})</h4>'
        document_header = f'<h4>Document {idx + 1}, (Doc: {doc.get("filename")})</h4>'
        html_content += document_header

        # If the document is an image
        if doc_type == "image":
            image_path = doc.get("source")
            try:
                with open(image_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                image_html = (
                    '<div style="border: 1px solid #ccc; padding: 10px; background-color: #f0f0f0; '
                    'width: 100%; margin-top: 20px;">'
                )
                image_html += (
                    f'<img src="data:image/jpeg;base64,{encoded_string}" alt="Image {doc.get("chunk_id")}" '
                    'style="width: 50%; height: auto;" />'
                )
                image_summary = f'<p><strong>Image Summary:</strong> {doc.get("page_content")}</p>'
                image_html += f'{image_summary}</div>'
                html_content += image_html
            except Exception as e:
                html_content += f'<div style="color: red; margin-top: 10px;">Could not open image at {image_path}: {e}</div>'

        # If the document is a table
        elif doc_type == "table":
            table_html = doc.get("source", "")
            if table_html:
                table_html = format_table_html(table_html)
                table_summary = f'<p><strong>Table Summary:</strong> {doc.get("page_content")}</p>'
                html_content += (
                    f'<div style="margin-top: 20px; border: 1px solid #ccc; padding: 10px; background-color: #f0f0f0;">'
                    f'{table_html}<br>{table_summary}</div>'
                )

        # If the document is plain text
        elif doc_type == "text":
            converted_doc_string = str(doc.get("page_content", "")).replace("\n", "<br>")
            html_content += (
                f'<div style="margin-top: 20px; padding: 10px; border: 1px solid #ccc; '
                f'background-color: #f0f0f0;">{converted_doc_string}</div>'
            )
        else:
            # Unknown type: just display raw content
            content = str(doc.get("page_content", "")).replace("\n", "<br>")
            html_content += (
                f'<div style="margin-top: 20px; padding: 10px; border: 1px solid #ccc; '
                f'background-color: #fff;">{content}</div>'
            )

    return html_content

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
                   temperature: float, stream: bool = True) -> Generator[Tuple[str, str], None, None]:
    backend = backend_url.strip() or DEFAULT_BACKEND
    q = (query or "").strip()
    if not q:
        yield "Please enter a non-empty query.", "<div>No docs.</div>"
        return

    ok, docs, err = call_reference(
        backend, q, num_chunks_post_rrf=num_chunks_post_rrf,
        num_docs_reranker=num_docs_reranker, use_reranker=use_reranker
    )
    if not ok:
        docs_html = f"<div style='color: red'>Error obtaining reference docs: {err}</div>"
        yield "", docs_html
        return

    # Create docs HTML using the provided renderer
    scores = [1.] * len(docs)
    docs_html = show_document_content(docs, scores) if docs else "<div>No reference docs returned.</div>"

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
        q_in = gr.Textbox(label="Question", placeholder="Write your question here...", lines=2)

        with gr.Row():
            with gr.Column():
                use_reranker = gr.Checkbox(label="Use reranker", value=True)
                num_chunks_post_rrf = gr.Slider(label="num_chunks_post_rrf", minimum=1, maximum=50, step=1, value=10)
                num_docs_reranker = gr.Slider(label="num_docs_reranker", minimum=1, maximum=20, step=1, value=3)
            with gr.Column():
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

        ask_btn.click(fn=ask_and_stream,
                      inputs=[backend_in, q_in, use_reranker, num_chunks_post_rrf, num_docs_reranker, max_tokens, temperature],
                      outputs=[answer_out, docs_out])

    return demo

if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=11000, share=False)
