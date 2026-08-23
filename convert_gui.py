import json
import os
import re
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from typing import Callable, Iterator

ROOT = Path(__file__).parent
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
WORKER_SCRIPT = ROOT / "marker_worker.py"
DEFAULT_OUTPUT = ROOT / "output"

# Matches a tqdm progress line on stderr, e.g.
#   "Recognizing Layout: 100%|██████| 3/10 [00:04<00:00, 4.07s/it]"
TQDM_RE = re.compile(r"^(?P<desc>.+?):\s+(?P<pct>\d+)%\|.*?\|\s*(?P<cur>\d+)/(?P<total>\d+)")
# marker logs this line when it finishes writing the document.
DONE_RE = re.compile(r"Saved markdown to (?P<path>.+)$")

MUTED = "#6b7280"
GLYPHS = {"pending": "•", "running": "▶", "done": "✓", "failed": "✗", "cancelled": "—"}


def iter_stderr(stream) -> Iterator[str]:
    """Yield logical lines from a byte stream, splitting on both \\r and \\n.

    tqdm redraws its bar with carriage returns, so plain line iteration would
    not flush intermediate updates. We split on either delimiter instead.
    """
    buf = b""
    while True:
        chunk = stream.read(1)
        if not chunk:
            break
        if chunk in (b"\n", b"\r"):
            if buf:
                yield buf.decode("utf-8", "replace")
                buf = b""
        else:
            buf += chunk
    if buf:
        yield buf.decode("utf-8", "replace")


class MarkerWorker:
    """Client for the persistent marker_worker.py process.

    The worker loads the models once and stays alive, so a batch pays that cost
    a single time instead of once per file. Its stderr is drained on a dedicated
    thread and parsed into progress events; requests and results travel as JSON
    lines over stdin/stdout.
    """

    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.device = "?"
        self.emit: Callable[[str, object], None] = lambda kind, payload: None
        self.tail: list[str] = []  # rolling stderr tail, for error reports
        self.lock = threading.Lock()

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self) -> tuple[bool, str]:
        """Spawn the worker and block until its models are loaded."""
        if self.alive:
            return True, self.device
        if not WORKER_SCRIPT.exists():
            return False, f"Worker script missing: {WORKER_SCRIPT}"

        python = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
        self.tail = []
        self.proc = subprocess.Popen(
            [python, str(WORKER_SCRIPT)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(os.environ, PYTHONUNBUFFERED="1"),
        )
        threading.Thread(target=self._drain_stderr, args=(self.proc,), daemon=True).start()

        handshake = self._read_reply()
        if not handshake.get("ready"):
            return False, handshake.get("error", "Worker failed to start")
        self.device = handshake.get("device", "?")
        return True, self.device

    def convert(self, pdf: str, out_dir: str) -> tuple[bool, str]:
        """Convert one PDF. Returns (ok, output path or error detail)."""
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        # Held across the reply too: one request may be in flight at a time.
        with self.lock:
            if not self.alive:
                return False, "Worker is not running."
            try:
                request = json.dumps({"pdf": str(pdf), "out_dir": str(out_dir)}) + "\n"
                self.proc.stdin.write(request.encode())
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                return False, f"Worker went away: {e}"

            reply = self._read_reply()

        if reply.get("ok"):
            return True, reply["out"]
        return False, reply.get("error") or "Worker stopped unexpectedly."

    def _read_reply(self) -> dict:
        """Read one JSON line from the worker; {} if it died.

        Binds the process locally: a concurrent stop() may clear self.proc while
        this call is parked in readline().
        """
        proc = self.proc
        if proc is None:
            return {}
        line = proc.stdout.readline()
        if not line:
            return {"error": "\n".join(self.tail[-12:]) or "Worker exited unexpectedly."}
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return {"error": f"Unexpected worker output: {line[:200]!r}"}

    def _drain_stderr(self, proc: subprocess.Popen) -> None:
        """Background thread: parse tqdm bars into progress events."""
        for line in iter_stderr(proc.stderr):
            line = line.strip()
            if not line:
                continue
            self.tail.append(line)
            del self.tail[:-40]

            m = TQDM_RE.match(line)
            if m:
                self.emit("progress", m.groupdict())
            elif DONE_RE.search(line):
                self.emit("status", "Finalizing…")

    def stop(self) -> None:
        """Kill the worker; the next conversion transparently starts a new one."""
        if self.alive:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None


class OutputPicker(ttk.Frame):
    """Label + entry + browse button for choosing the output folder."""

    def __init__(self, master: tk.Misc, variable: tk.StringVar):
        super().__init__(master)
        self.columnconfigure(0, weight=1)
        self.variable = variable
        ttk.Label(self, text="Output folder:").grid(row=0, column=0, sticky="w", pady=(0, 4))
        ttk.Entry(self, textvariable=variable).grid(row=1, column=0, sticky="ew")
        ttk.Button(self, text="Browse…", command=self.pick).grid(row=1, column=1, padx=(8, 0))

    def pick(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.variable.set(path)


class SingleTab(ttk.Frame):
    """Convert one PDF at a time."""

    def __init__(self, master: tk.Misc, app: "App"):
        super().__init__(master, padding=16)
        self.app = app
        self.columnconfigure(0, weight=1)

        self.pdf_path = tk.StringVar(value="No file selected")
        self.out_path = tk.StringVar(value=str(DEFAULT_OUTPUT))
        self.events: queue.Queue = queue.Queue()

        ttk.Label(self, text="PDF file:").grid(row=0, column=0, sticky="w", pady=(0, 4))
        row = ttk.Frame(self)
        row.grid(row=1, column=0, sticky="ew")
        row.columnconfigure(0, weight=1)
        ttk.Entry(row, textvariable=self.pdf_path, state="readonly").grid(row=0, column=0, sticky="ew")
        ttk.Button(row, text="Browse…", command=self.pick_pdf).grid(row=0, column=1, padx=(8, 0))

        OutputPicker(self, self.out_path).grid(row=2, column=0, sticky="ew", pady=(12, 0))

        self.btn = ttk.Button(self, text="Convert", command=self.convert)
        self.btn.grid(row=3, column=0, pady=(20, 8))

        self.progress = ttk.Progressbar(self, length=420, mode="determinate", maximum=100)
        self.progress.grid(row=4, column=0, sticky="ew", pady=(4, 6))

        self.status = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status, foreground=MUTED).grid(row=5, column=0, sticky="w")

    def pick_pdf(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if path:
            self.pdf_path.set(path)

    def convert(self) -> None:
        pdf = self.pdf_path.get()
        if pdf == "No file selected" or not Path(pdf).exists():
            messagebox.showerror("Error", "Please select a valid PDF file.")
            return
        if not self.app.claim_worker():
            return

        self.btn.config(state="disabled")
        self.progress.config(mode="indeterminate")
        self.progress.start(12)  # spin until the first stage reports a count
        self.status.set("Loading models…")
        # Read the Tk variable here: only the main thread may touch Tk state.
        threading.Thread(target=self._run, args=(pdf, self.out_path.get()), daemon=True).start()
        self.after(100, self._drain_events)

    def _run(self, pdf: str, out_dir: str) -> None:
        """Worker thread: convert and push the outcome to the queue."""
        worker = self.app.worker
        worker.emit = lambda kind, payload: self.events.put((kind, payload))
        started, info = worker.start()
        if not started:
            self.events.put(("error", info))
            return
        self.events.put(("status", f"Converting on {info.upper()}…"))
        ok, detail = worker.convert(pdf, out_dir)
        self.events.put(("done" if ok else "error", detail))

    def _drain_events(self) -> None:
        """Main thread: apply queued updates to the widgets."""
        running = True
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "progress":
                    self._apply_progress(payload)
                elif kind == "status":
                    self.status.set(payload)
                elif kind == "done":
                    self._finish(f"Done → {payload}", success=True)
                    running = False
                elif kind == "error":
                    self._finish("Error — see details.", success=False, detail=payload)
                    running = False
        except queue.Empty:
            pass
        if running:
            self.after(100, self._drain_events)

    def _apply_progress(self, info: dict) -> None:
        if str(self.progress.cget("mode")) == "indeterminate":
            self.progress.stop()
            self.progress.config(mode="determinate")
        pct = int(info["pct"])
        cur, total = info["cur"], info["total"]
        self.progress["value"] = pct
        unit = "page" if total == "1" else "pages"
        self.status.set(f"{info['desc']} — {cur}/{total} {unit} ({pct}%)")

    def _finish(self, message: str, success: bool, detail: str | None = None) -> None:
        self.progress.stop()
        self.progress.config(mode="determinate")
        self.progress["value"] = 100 if success else 0
        self.status.set(message)
        self.btn.config(state="normal")
        self.app.release_worker()
        if not success:
            messagebox.showerror("Conversion failed", detail or "Unknown error")


class BatchTab(ttk.Frame):
    """Queue up many PDFs and convert them one after another."""

    def __init__(self, master: tk.Misc, app: "App"):
        super().__init__(master, padding=16)
        self.app = app
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self.out_path = tk.StringVar(value=str(DEFAULT_OUTPUT))
        self.events: queue.Queue = queue.Queue()
        self.files: list[Path] = []
        self.states: dict[str, str] = {}  # path → pending/running/done/failed/cancelled
        self.errors: dict[str, str] = {}  # path → stderr tail
        self.cancel = threading.Event()

        ttk.Label(self, text="PDF files:").grid(row=0, column=0, sticky="w", pady=(0, 4))

        listbox_row = ttk.Frame(self)
        listbox_row.grid(row=1, column=0, sticky="nsew")
        listbox_row.columnconfigure(0, weight=1)
        listbox_row.rowconfigure(0, weight=1)
        self.listbox = tk.Listbox(listbox_row, selectmode="extended", height=10, activestyle="none")
        self.listbox.grid(row=0, column=0, sticky="nsew")
        self.listbox.bind("<Double-Button-1>", self._show_error)
        scroll = ttk.Scrollbar(listbox_row, orient="vertical", command=self.listbox.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.listbox.config(yscrollcommand=scroll.set)

        buttons = ttk.Frame(self)
        buttons.grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.list_buttons = [
            ttk.Button(buttons, text="Add Files…", command=self.add_files),
            ttk.Button(buttons, text="Add Folder…", command=self.add_folder),
            ttk.Button(buttons, text="Remove Selected", command=self.remove_selected),
            ttk.Button(buttons, text="Clear", command=self.clear),
        ]
        for i, b in enumerate(self.list_buttons):
            b.grid(row=0, column=i, padx=(0, 6))

        OutputPicker(self, self.out_path).grid(row=3, column=0, sticky="ew", pady=(14, 0))

        actions = ttk.Frame(self)
        actions.grid(row=4, column=0, pady=(18, 8))
        self.btn = ttk.Button(actions, text="Convert All", command=self.convert)
        self.btn.grid(row=0, column=0, padx=(0, 8))
        self.stop_btn = ttk.Button(actions, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.grid(row=0, column=1)

        ttk.Label(self, text="Overall", foreground=MUTED).grid(row=5, column=0, sticky="w")
        self.overall = ttk.Progressbar(self, length=420, mode="determinate", maximum=100)
        self.overall.grid(row=6, column=0, sticky="ew", pady=(2, 8))

        ttk.Label(self, text="Current file", foreground=MUTED).grid(row=7, column=0, sticky="w")
        self.progress = ttk.Progressbar(self, length=420, mode="determinate", maximum=100)
        self.progress.grid(row=8, column=0, sticky="ew", pady=(2, 6))

        self.status = tk.StringVar(value="No files queued")
        ttk.Label(self, textvariable=self.status, foreground=MUTED).grid(row=9, column=0, sticky="w")

    # ---- file list management -------------------------------------------------

    def add_files(self) -> None:
        paths = filedialog.askopenfilenames(filetypes=[("PDF files", "*.pdf")])
        self._add([Path(p) for p in paths])

    def add_folder(self) -> None:
        folder = filedialog.askdirectory()
        if folder:
            self._add(sorted(Path(folder).rglob("*.pdf")))

    def _add(self, paths: list[Path]) -> None:
        known = {str(f) for f in self.files}
        fresh: list[Path] = []
        for p in paths:
            if str(p) not in known:
                known.add(str(p))
                fresh.append(p)
        self.files = self.files + fresh
        self.states = {**self.states, **{str(p): "pending" for p in fresh}}
        self._render()

    def remove_selected(self) -> None:
        doomed = {self.files[i] for i in self.listbox.curselection()}
        self.files = [f for f in self.files if f not in doomed]
        self._render()

    def clear(self) -> None:
        self.files = []
        self.states = {}
        self.errors = {}
        self._render()

    def _render(self) -> None:
        self.listbox.delete(0, tk.END)
        for f in self.files:
            glyph = GLYPHS[self.states.get(str(f), "pending")]
            self.listbox.insert(tk.END, f"  {glyph}  {f.name}")
        if not self.app.busy:
            n = len(self.files)
            self.status.set("No files queued" if not n else f"{n} file{'s' if n != 1 else ''} queued")

    def _set_state(self, path: Path, state: str) -> None:
        self.states = {**self.states, str(path): state}
        index = self.files.index(path)
        self.listbox.delete(index)
        self.listbox.insert(index, f"  {GLYPHS[state]}  {path.name}")

    def _show_error(self, _event: tk.Event) -> None:
        selection = self.listbox.curselection()
        if not selection:
            return
        detail = self.errors.get(str(self.files[selection[0]]))
        if detail:
            messagebox.showerror("Conversion failed", detail)

    # ---- conversion -----------------------------------------------------------

    def convert(self) -> None:
        missing = [f for f in self.files if not f.exists()]
        if not self.files:
            messagebox.showerror("Error", "Add at least one PDF file.")
            return
        if missing:
            messagebox.showerror("Error", "These files no longer exist:\n" + "\n".join(m.name for m in missing))
            return
        if not self.app.claim_worker():
            return

        self.errors = {}
        self.states = {str(f): "pending" for f in self.files}
        self._render()
        self.cancel.clear()
        self.btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        for b in self.list_buttons:
            b.config(state="disabled")
        self.overall["value"] = 0
        self.status.set("Loading models…")
        threading.Thread(target=self._run, args=(list(self.files), self.out_path.get()), daemon=True).start()
        self.after(100, self._drain_events)

    def stop(self) -> None:
        """Cancel the batch. This kills the worker, so the models reload on the
        next run — an acceptable trade for interrupting the in-flight file."""
        self.cancel.set()
        self.status.set("Stopping…")
        self.stop_btn.config(state="disabled")
        self.app.worker.stop()

    def _run(self, files: list[Path], out_dir: str) -> None:
        """Worker thread: convert each queued PDF in turn, models loaded once."""
        worker = self.app.worker
        worker.emit = lambda kind, payload: self.events.put((kind, payload))
        started, info = worker.start()
        if not started:
            self.events.put(("fatal", info))
            return
        self.events.put(("status", f"Models ready on {info.upper()} — starting…"))

        succeeded = 0
        for i, pdf in enumerate(files):
            if self.cancel.is_set():
                self.events.put(("file_state", (pdf, "cancelled", "")))
            else:
                self.events.put(("file_start", (pdf, i, len(files))))
                ok, detail = worker.convert(str(pdf), out_dir)
                if ok:
                    succeeded += 1
                    self.events.put(("file_state", (pdf, "done", detail)))
                else:
                    state = "cancelled" if self.cancel.is_set() else "failed"
                    self.events.put(("file_state", (pdf, state, detail)))
            self.events.put(("overall", (i + 1) / len(files) * 100))
        self.events.put(("batch_done", (succeeded, len(files))))

    def _drain_events(self) -> None:
        running = True
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "progress":
                    self._apply_progress(payload)
                elif kind == "status":
                    self.status.set(payload)
                elif kind == "file_start":
                    pdf, i, total = payload
                    self._set_state(pdf, "running")
                    self.listbox.see(self.files.index(pdf))
                    self.progress["value"] = 0
                    self.status.set(f"[{i + 1}/{total}] {pdf.name} — loading…")
                elif kind == "file_state":
                    pdf, state, detail = payload
                    self._set_state(pdf, state)
                    if state == "failed":
                        self.errors = {**self.errors, str(pdf): detail}
                elif kind == "overall":
                    self.overall["value"] = payload
                elif kind == "fatal":
                    self._finish(0, len(self.files))
                    messagebox.showerror("Could not start marker", payload)
                    running = False
                elif kind == "batch_done":
                    self._finish(*payload)
                    running = False
        except queue.Empty:
            pass
        if running:
            self.after(100, self._drain_events)

    def _apply_progress(self, info: dict) -> None:
        pct = int(info["pct"])
        cur, total = info["cur"], info["total"]
        self.progress["value"] = pct
        unit = "page" if total == "1" else "pages"
        prefix = self.status.get().split(" — ")[0]
        self.status.set(f"{prefix} — {info['desc']} {cur}/{total} {unit} ({pct}%)")

    def _finish(self, succeeded: int, total: int) -> None:
        self.progress["value"] = 0
        self.overall["value"] = 100 if succeeded == total else self.overall["value"]
        self.btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        for b in self.list_buttons:
            b.config(state="normal")
        self.app.release_worker()

        failed = sum(1 for s in self.states.values() if s == "failed")
        cancelled = sum(1 for s in self.states.values() if s == "cancelled")
        parts = [f"{succeeded}/{total} converted"]
        if failed:
            parts.append(f"{failed} failed")
        if cancelled:
            parts.append(f"{cancelled} skipped")
        self.status.set(" · ".join(parts) + (" — double-click a ✗ row for details." if failed else ""))
        if failed:
            names = "\n".join(Path(p).name for p, s in self.states.items() if s == "failed")
            messagebox.showerror("Some conversions failed", f"{failed} file(s) failed:\n\n{names}")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PDF → Markdown with Marker")
        self.minsize(560, 300)
        self.busy = False  # one conversion at a time — the worker is single-threaded
        self.worker = MarkerWorker()

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=12, pady=(12, 0))
        notebook.add(SingleTab(notebook, self), text="Single")
        notebook.add(BatchTab(notebook, self), text="Batch")

        ttk.Label(self, text=self._tuning_summary(), foreground=MUTED).pack(
            anchor="w", padx=16, pady=(6, 10)
        )
        self.protocol("WM_DELETE_WINDOW", self.close)

    @staticmethod
    def _tuning_summary() -> str:
        """One-line note about the hardware tuning the worker will apply."""
        import marker_worker

        if not marker_worker.is_apple_silicon():
            return "Models stay loaded between files."
        ram = marker_worker.unified_memory_gb()
        batch = marker_worker.apple_silicon_env().get("RECOGNITION_BATCH_SIZE", "default")
        return f"Apple Silicon · {ram} GB unified memory · recognition batch {batch} · models stay loaded between files"

    def claim_worker(self) -> bool:
        """Guard against running two conversions at once; warns the user if busy."""
        if not WORKER_SCRIPT.exists():
            messagebox.showerror("Error", f"Worker script not found at:\n{WORKER_SCRIPT}")
            return False
        if self.busy:
            messagebox.showinfo("Busy", "A conversion is already running. Wait for it to finish.")
            return False
        self.busy = True
        return True

    def release_worker(self) -> None:
        self.busy = False

    def close(self) -> None:
        """Tear down the model process before the window goes away."""
        self.worker.stop()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
