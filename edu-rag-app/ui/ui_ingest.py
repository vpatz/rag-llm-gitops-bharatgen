import gradio as gr
from pathlib import Path
import importlib
import subprocess
import os
import shutil
import zipfile
import sys
from typing import Generator, Tuple, List
import threading
import queue
import time
import json

SETTINGS_PATH = Path(__file__).resolve().parent.parent / "src/settings.json"
with open(SETTINGS_PATH, 'r') as f:
    data = json.load(f)
db_prefix = str(data.get('milvus_db_prefix'))
collection_name = str(data.get('milvus_collection_name'))
db_ui_port = int(data.get("db_ui_port"))

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


# Minimal logger that collects lines for UI
def _append_log(buf: List[str], msg: str):
    if not msg.endswith("\n"):
        msg = msg + "\n"
    buf.append(msg)


def _save_and_unzip(uploaded_zip, out_path: str, log_buf: List[str]) -> str:
    os.makedirs(out_path, exist_ok=True)
    raw_dir = os.path.join(out_path, "raw")
    if os.path.exists(raw_dir):
        _append_log(log_buf, f"Removing existing raw directory: {raw_dir}")
        shutil.rmtree(raw_dir)
    os.makedirs(raw_dir, exist_ok=True)

    zip_save_path = os.path.join(out_path, "uploaded.zip")
    _append_log(log_buf, f"Saving uploaded ZIP to {zip_save_path}")
    with open(zip_save_path, "wb") as f_out:
        if hasattr(uploaded_zip, "read"):
            uploaded_zip.seek(0)
            shutil.copyfileobj(uploaded_zip, f_out)
        else:
            shutil.copy(uploaded_zip, zip_save_path)

    _append_log(log_buf, f"Extracting into {raw_dir}")
    try:
        with zipfile.ZipFile(zip_save_path, "r") as zf:
            for member in zf.namelist():
                if member.startswith("__MACOSX") or member.endswith("/"):
                    continue
                dest_path = os.path.normpath(os.path.join(raw_dir, member))
                if not dest_path.startswith(os.path.abspath(raw_dir)):
                    _append_log(log_buf, f"Skipping suspicious member: {member}")
                    continue
                dest_dir = os.path.dirname(dest_path)
                if dest_dir and not os.path.exists(dest_dir):
                    os.makedirs(dest_dir, exist_ok=True)
                with zf.open(member) as src, open(dest_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    except zipfile.BadZipFile:
        raise
    _append_log(log_buf, "Unzip complete.")
    return raw_dir

def stream_cli_process_generator(module_name: str, argv: List[str], cwd: str, logs: List[str], poll_interval: float = 0.1):
    """
    Generator that runs `python -u -m <module_name> ...` and yields (logs_str, status) tuples
    whenever new lines are available. Also prints lines to terminal immediately.

    - module_name: "src.ingest.cli"
    - argv: list of tokens AFTER the module, e.g. ["ingest", "--path", raw_dir] or ["clean-db"]
    - cwd: working directory for subprocess (project root)
    - logs: list buffer to append lines to (shared)
    """

    # Build command
    cmd = [sys.executable, "-u", "-m", module_name] + argv

    _append_log(logs, f"Starting subprocess: {' '.join(cmd)} (cwd={cwd})")
    print(f"[CLI RUN] {' '.join(cmd)} (cwd={cwd})", flush=True)

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    q: "queue.Queue[tuple[str,str]]" = queue.Queue()

    def _reader(pipe, tag):
        try:
            for line in iter(pipe.readline, ""):
                if line is None:
                    break
                # strip trailing newline but preserve content
                text = line.rstrip("\n")
                q.put((tag, text))
        except Exception as e:
            q.put(("stderr", f"Reader error: {e}"))
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    t_out = threading.Thread(target=_reader, args=(proc.stdout, "stdout"), daemon=True)
    t_err = threading.Thread(target=_reader, args=(proc.stderr, "stderr"), daemon=True)
    t_out.start()
    t_err.start()

    status = "Running"
    last_yield_time = time.time()
    # yield initial state
    yield ("".join(logs), status)

    # Loop until process exits and queue drained
    while True:
        got_any = False
        try:
            # drain all available lines without blocking for long
            while True:
                tag, line = q.get_nowait()
                got_any = True
                if tag == "stdout":
                    # append to logs and print to terminal
                    _append_log(logs, f"=== STDOUT === {line}")
                    print(f"=== STDOUT === {line}", flush=True)
                else:
                    _append_log(logs, f"=== STDERR === {line}")
                    print(f"=== STDERR === {line}", flush=True)
        except queue.Empty:
            pass

        # if new lines arrived, yield updated logs so Gradio refreshes
        if got_any:
            yield ("".join(logs), status)

        # process finished?
        if proc.poll() is not None:
            # drain anything left (blocking until queue empty with small timeout)
            while True:
                try:
                    tag, line = q.get(timeout=0.1)
                    if tag == "stdout":
                        _append_log(logs, line)
                        print(line, flush=True)
                    else:
                        _append_log(logs, f"=== STDERR === {line}")
                        print(f"=== STDERR === {line}", flush=True)
                    # yield after each draining line
                    yield ("".join(logs), status)
                except queue.Empty:
                    break
            break

        time.sleep(poll_interval)

    # final status / exit code
    rc = proc.returncode
    if rc == 0:
        status = "Done"
        _append_log(logs, f"Subprocess finished with exit code 0")
        print(f"[CLI FINISHED] exit code 0", flush=True)
    else:
        status = f"Failed (exit {rc})"
        _append_log(logs, f"Subprocess finished with exit code {rc}")
        print(f"[CLI FINISHED] exit code {rc}", flush=True)

    # final yield with full logs and final status
    yield ("".join(logs), status)
    return

def _estimate_progress_from_logs(logs: List[str]) -> int:
    """
    Very rough progress estimator based on log keywords.
    Keeps UI feeling alive even if CLI doesn't emit %.
    """
    joined = "\n".join(logs).lower()

    if "saving uploaded zip" in joined:
        return 5
    if "extracting into" in joined:
        return 10
    if "processing file" in joined or "converting" in joined:
        return 30
    if "creating chunks" in joined:
        return 55
    if "embedding" in joined:
        return 75
    if "insert" in joined:
        return 90
    if "finished with exit code 0" in joined:
        return 100
    return 15

# ---------- Gradio generator ----------
def run_action(uploaded_zip, choice) -> Generator[Tuple[str, str], None, None]:
    logs: List[str] = []
    yield ("", "Starting", 0)

    module_name = "src.ingest.cli"

    # ---------------- CLEAN DB (no file required) ----------------
    if choice == "clean-db":
        _append_log(logs, "Running clean-db (no upload required).")
        yield ("".join(logs), "Cleaning database...", 10)

        for logs_snapshot, status in stream_cli_process_generator(
            module_name,
            ["clean-db"],
            str(proj_root),
            logs
        ):
            yield (logs_snapshot, status, _estimate_progress_from_logs(logs))

        yield ("".join(logs), "Database cleaned", 100)
        return

    if choice == "ingest" and uploaded_zip is None:
        _append_log(logs, "No ZIP uploaded for ingestion.")
        yield ("".join(logs), "No file", 0)
        return

    # Import setup_docs_dir from src.common.misc_utils (package import)
    out_path = os.path.abspath(f"{db_prefix}_{collection_name}_docs")
    try:
        common_mod = importlib.import_module("src.common.misc_utils")
        setup_fn = getattr(common_mod, "setup_docs_dir", None)
        if callable(setup_fn):
            try:
                out_path = setup_fn(f"{db_prefix}_{collection_name}_docs")
                _append_log(logs, f"setup_docs_dir returned: {out_path}")
            except Exception as e:
                _append_log(logs, f"setup_docs_dir() raised: {e}")
                out_path = os.path.abspath(f"{db_prefix}_{collection_name}_docs")
                _append_log(logs, f"Falling back to {out_path}")
        else:
            _append_log(logs, "setup_docs_dir not found in src.common.misc_utils; using fallback raw")
    except Exception as e:
        _append_log(logs, f"Could not import src.common.misc_utils: {e}")
        _append_log(logs, "Using fallback out_path")
        out_path = os.path.abspath(f"{db_prefix}_{collection_name}_docs")

    yield ("".join(logs), f"Using out_path: {out_path}", _estimate_progress_from_logs(logs))
    yield ("".join(logs), "Saving and extracting ZIP...", _estimate_progress_from_logs(logs))

    if choice == "ingest":
        yield ("".join(logs), "Saving and extracting ZIP...", _estimate_progress_from_logs(logs))

        try:
            raw_dir = _save_and_unzip(uploaded_zip, out_path, logs)
        except zipfile.BadZipFile:
            _append_log(logs, "Uploaded file is not a valid ZIP.")
            yield ("".join(logs), "Bad ZIP", 0)
            return

        cli_argv = ["ingest", "--path", raw_dir]

    yield ("".join(logs), f"Running CLI module via subprocess: python -m src.ingest.cli {' '.join(cli_argv)}", _estimate_progress_from_logs(logs))

    for logs_snapshot, status in stream_cli_process_generator(module_name, cli_argv, str(proj_root), logs):
        yield (logs_snapshot, status, _estimate_progress_from_logs(logs))

    yield ("".join(logs), "Finished", 100)


# ---------- Gradio UI ----------
def build_ui():
    with gr.Blocks(title="Vector-DB Utility", analytics_enabled=False) as demo:
        gr.Markdown("Upload a ZIP. It will be extracted and ingested. You may delete existing DB as well using this utility.")

        with gr.Row():
            zip_in = gr.File(label="ZIP file", file_types=[".zip"])
            action = gr.Radio(choices=["ingest", "clean-db"], value="ingest", label="Action")

        run_btn = gr.Button("Run")

        progress_bar = gr.Slider(label="Progress", minimum=0, maximum=100, value=0, interactive=False)
        status_box = gr.Textbox(label="Status", lines=1, interactive=False)
        log_box = gr.Textbox(label="Logs", lines=20, interactive=False)

        run_btn.click(
            fn=run_action,
            inputs=[zip_in, action],
            outputs=[log_box, status_box, progress_bar]
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.queue()  # enables background job handling
    app.launch(server_name="0.0.0.0", server_port=db_ui_port)
