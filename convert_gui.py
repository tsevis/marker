import json
import os
import re
import queue
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from typing import Callable, Iterator

from marker.image_export import ImageExportSettings, scaled_image_size
from marker_textlayer import TextLayerReport, inspect_text_layer

ROOT = Path(__file__).parent
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
WORKER_SCRIPT = ROOT / "marker_worker.py"
DEFAULT_OUTPUT = ROOT / "output"
SETTINGS_FILE = ROOT / ".marker-gui-settings.json"

# Matches a tqdm progress line on stderr, e.g.
#   "Recognizing Layout: 100%|██████| 3/10 [00:04<00:00, 4.07s/it]"
TQDM_RE = re.compile(r"^(?P<desc>.+?):\s+(?P<pct>\d+)%\|.*?\|\s*(?P<cur>\d+)/(?P<total>\d+)")
# marker logs this line when it finishes writing the document.
DONE_RE = re.compile(r"Saved markdown to (?P<path>.+)$")

MUTED = "#6b7280"
SKIP_OCR_LABEL = "Skip OCR and use the PDF's own text (faster)"
BATCH_SKIP_OCR_LABEL = "Skip OCR for PDFs that have their own text (faster)"
GLYPHS = {"pending": "•", "running": "▶", "done": "✓", "failed": "✗", "cancelled": "—"}


def load_image_settings(path: Path) -> tuple[ImageExportSettings, str]:
    try:
        return ImageExportSettings.from_dict(json.loads(path.read_text())), ""
    except FileNotFoundError:
        return ImageExportSettings(), ""
    except (OSError, ValueError) as exc:
        return ImageExportSettings(), f"Could not load saved settings; using defaults. {exc}"


def save_image_settings(path: Path, settings: ImageExportSettings) -> None:
    """Replace preferences atomically so a stopped GUI cannot truncate them."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(settings.to_dict(), stream, indent=2)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def text_layer_report(pdf: str) -> TextLayerReport | None:
    """The PDF's text-layer report, or None when the file cannot be inspected.

    None means "do not offer to skip OCR": an unreadable file is left for the
    conversion itself to report, with marker's own error message.
    """
    try:
        return inspect_text_layer(pdf)
    except Exception:  # noqa: BLE001 - PDFium raises its own types per failure
        return None


def can_skip_ocr(pdf: str) -> bool:
    """True only when the PDF is readable and has a text layer."""
    report = text_layer_report(pdf)
    return report is not None and report.suggest_skip_ocr


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

    def convert(
        self,
        pdf: str,
        out_dir: str,
        image_settings: ImageExportSettings | None = None,
        skip_ocr: bool = False,
    ) -> tuple[bool, str]:
        """Convert one PDF. Returns (ok, output path or error detail)."""
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        # Held across the reply too: one request may be in flight at a time.
        with self.lock:
            if not self.alive:
                return False, "Worker is not running."
            try:
                request = json.dumps({
                    "pdf": str(pdf),
                    "out_dir": str(out_dir),
                    "image_settings": (image_settings or ImageExportSettings()).to_dict(),
                    "skip_ocr": bool(skip_ocr),
                }) + "\n"
                self.proc.stdin.write(request.encode())
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                return False, f"Worker went away: {e}"

            reply = self._read_reply()

        if reply.get("ok"):
            return True, reply["out"]
        return False, reply.get("error") or "Worker stopped unexpectedly."

    def _read_reply(self) -> dict:
        """Read the worker's next reply, forwarding any progress it reports first.

        The worker interleaves {"progress": ...} lines with the reply to a
        request, so this consumes them until an actual reply arrives.

        Binds the process locally: a concurrent stop() may clear self.proc while
        this call is parked in readline().
        """
        proc = self.proc
        if proc is None:
            return {}
        while True:
            line = proc.stdout.readline()
            if not line:
                return {
                    "error": "\n".join(self.tail[-12:])
                    or "Worker exited unexpectedly."
                }
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                return {"error": f"Unexpected worker output: {line[:200]!r}"}
            if not isinstance(message, dict):
                return {"error": f"Unexpected worker output: {line[:200]!r}"}
            progress = message.get("progress")
            if progress is None:
                return message
            self.emit("progress", progress)

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


class SettingsTab(ttk.Frame):
    """Export preferences shared by single-file and batch conversion."""

    def __init__(self, master: tk.Misc, app: "App", load_warning: str = ""):
        super().__init__(master, padding=16)
        self.app = app
        self.columnconfigure(0, weight=1)
        options = app.image_settings
        self.dpi = tk.StringVar(value=str(options.dpi))
        self.scale = tk.StringVar(value=str(options.scale_percent))
        self.limit_enabled = tk.BooleanVar(value=bool(options.max_dimension))
        self.max_dimension = tk.StringVar(value=str(options.max_dimension or 2400))
        self.preview = tk.StringVar()
        self.status = tk.StringVar(value=load_warning)

        ttk.Label(self, text="Photo export", font=("TkDefaultFont", 15, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            self, text="Shared by Single and Batch. Changes apply to the next conversion.",
            foreground=MUTED,
        ).grid(row=1, column=0, sticky="w", pady=(4, 18))

        fields = ttk.Frame(self)
        fields.grid(row=2, column=0, sticky="ew")
        fields.columnconfigure(1, weight=1)
        ttk.Label(fields, text="Resolution (DPI)").grid(row=0, column=0, sticky="w", padx=(0, 20))
        ttk.Combobox(
            fields, textvariable=self.dpi, values=(72, 96, 150, 192, 300, 450, 600), width=12,
        ).grid(row=0, column=1, sticky="w")
        ttk.Label(
            fields, text="Choose a preset or enter 72–600. Default: 192 DPI.", foreground=MUTED,
        ).grid(row=1, column=1, sticky="w", pady=(3, 14))

        ttk.Label(fields, text="Scale (%)").grid(row=2, column=0, sticky="w")
        ttk.Combobox(
            fields, textvariable=self.scale, values=(25, 50, 75, 100, 150, 200, 400), width=12,
        ).grid(row=2, column=1, sticky="w")
        ttk.Label(
            fields, text="10–400%. 100% keeps the dimensions rendered at your DPI.",
            foreground=MUTED,
        ).grid(row=3, column=1, sticky="w", pady=(3, 14))

        ttk.Checkbutton(
            fields, text="Limit longest edge", variable=self.limit_enabled,
        ).grid(row=4, column=0, sticky="w", padx=(0, 20))
        limit_row = ttk.Frame(fields)
        limit_row.grid(row=4, column=1, sticky="w")
        self.limit_entry = ttk.Spinbox(
            limit_row, textvariable=self.max_dimension, from_=1, to=20000, increment=100, width=12,
        )
        self.limit_entry.pack(side="left")
        ttk.Label(limit_row, text="pixels").pack(side="left", padx=(8, 0))
        ttk.Label(
            fields, text="Shrinks images that exceed the limit; keeps their proportions.", foreground=MUTED,
        ).grid(row=5, column=1, sticky="w", pady=(3, 14))

        example = ttk.LabelFrame(self, text="Size example", padding=12)
        example.grid(row=3, column=0, sticky="ew", pady=(2, 14))
        ttk.Label(example, text="A photo measuring 4 × 3 inches on the PDF page:").pack(anchor="w")
        ttk.Label(example, textvariable=self.preview, font=("TkDefaultFont", 13, "bold")).pack(
            anchor="w", pady=(6, 0)
        )
        ttk.Label(
            self, wraplength=630, foreground=MUTED,
            text="Photos are cropped from PDF pages and saved as JPEG. Higher DPI can retain more "
                 "source detail and uses more time and memory. Scaling above 100% enlarges pixels; "
                 "it does not recover original detail. OCR resolution stays the same.",
        ).grid(row=4, column=0, sticky="w", pady=(0, 14))

        buttons = ttk.Frame(self)
        buttons.grid(row=5, column=0, sticky="w")
        ttk.Button(buttons, text="Save settings", command=self.apply).pack(side="left")
        ttk.Button(buttons, text="Restore defaults", command=self.restore_defaults).pack(side="left", padx=(8, 0))
        ttk.Label(
            self, text="Settings are remembered when saved or when a conversion starts.", foreground=MUTED,
        ).grid(row=6, column=0, sticky="w", pady=(8, 0))
        ttk.Label(self, textvariable=self.status, wraplength=630).grid(
            row=7, column=0, sticky="w", pady=(6, 0)
        )
        for variable in (self.dpi, self.scale, self.limit_enabled, self.max_dimension):
            variable.trace_add("write", self._changed)
        self._update_preview()

    def read(self) -> ImageExportSettings:
        try:
            dpi, scale = int(self.dpi.get()), int(self.scale.get())
            limit = int(self.max_dimension.get()) if self.limit_enabled.get() else 0
        except ValueError:
            raise ValueError("Enter whole numbers for resolution, scale and the pixel limit.") from None
        if self.limit_enabled.get() and limit == 0:
            raise ValueError("The longest-edge limit must be at least 1 pixel, or turn the limit off.")
        return ImageExportSettings(dpi, scale, limit)

    def _update_preview(self) -> None:
        self.limit_entry.config(state="normal" if self.limit_enabled.get() else "disabled")
        try:
            options = self.read()
            width, height = scaled_image_size(
                (4 * options.dpi, 3 * options.dpi), options.scale_percent, options.max_dimension
            )
            self.preview.set(f"Approximately {width:,} × {height:,} pixels")
        except ValueError:
            self.preview.set("Enter valid settings to preview the size.")

    def _changed(self, *_args) -> None:
        self.status.set("Changes will be saved when you start a conversion or click Save settings.")
        self._update_preview()

    def apply(self) -> ImageExportSettings | None:
        try:
            options = self.read()
        except ValueError as exc:
            self.app.notebook.select(self)
            messagebox.showerror("Invalid export settings", str(exc), parent=self)
            return None
        try:
            save_image_settings(self.app.settings_path, options)
        except OSError as exc:
            messagebox.showerror("Could not save settings", str(exc), parent=self)
            return None
        self.app.image_settings = options
        self.app.export_summary.set(options.summary())
        self.status.set("Settings saved. They apply to the next conversion or batch.")
        return options

    def restore_defaults(self) -> None:
        options = ImageExportSettings()
        self.dpi.set(str(options.dpi))
        self.scale.set(str(options.scale_percent))
        self.limit_enabled.set(False)
        self.max_dimension.set("2400")
        self.apply()


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

        # Offered only once the chosen PDF is found to carry its own text.
        self.skip_ocr = tk.BooleanVar(value=False)
        self.ocr_hint = tk.StringVar(value="")
        ocr_row = ttk.Frame(self)
        ocr_row.grid(row=3, column=0, sticky="w", pady=(12, 0))
        self.skip_ocr_box = ttk.Checkbutton(
            ocr_row, text=SKIP_OCR_LABEL, variable=self.skip_ocr, state="disabled"
        )
        self.skip_ocr_box.grid(row=0, column=0, sticky="w")
        ttk.Label(ocr_row, textvariable=self.ocr_hint, foreground=MUTED).grid(
            row=1, column=0, sticky="w"
        )

        self.btn = ttk.Button(self, text="Convert", command=self.convert)
        self.btn.grid(row=4, column=0, pady=(20, 8))

        self.progress = ttk.Progressbar(self, length=420, mode="determinate", maximum=100)
        self.progress.grid(row=5, column=0, sticky="ew", pady=(4, 6))

        self.status = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status, foreground=MUTED).grid(row=6, column=0, sticky="w")

    def pick_pdf(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if path:
            self.pdf_path.set(path)
            self._offer_skip_ocr(text_layer_report(path))

    def _offer_skip_ocr(self, report: TextLayerReport | None) -> None:
        """Enable the Skip OCR choice only for a PDF with a text layer.

        It is never ticked for the user: the hint is the suggestion.
        """
        offered = report is not None and report.suggest_skip_ocr
        self.skip_ocr.set(False)
        self.skip_ocr_box.config(state="normal" if offered else "disabled")
        self.ocr_hint.set("" if report is None else report.hint())

    def convert(self) -> None:
        pdf = self.pdf_path.get()
        if pdf == "No file selected" or not Path(pdf).exists():
            messagebox.showerror("Error", "Please select a valid PDF file.")
            return
        options = self.app.settings_tab.apply()
        if options is None:
            return
        if not self.app.claim_worker():
            return

        self.btn.config(state="disabled")
        self.progress.config(mode="indeterminate")
        self.progress.start(12)  # spin until the first stage reports a count
        self.status.set("Loading models…")
        # Read the Tk variable here: only the main thread may touch Tk state.
        threading.Thread(
            target=self._run,
            args=(pdf, self.out_path.get(), options, self.skip_ocr.get()),
            daemon=True,
        ).start()
        self.after(100, self._drain_events)

    def _run(
        self, pdf: str, out_dir: str, options: ImageExportSettings, skip_ocr: bool = False
    ) -> None:
        """Worker thread: convert and push the outcome to the queue."""
        worker = self.app.worker
        worker.emit = lambda kind, payload: self.events.put((kind, payload))
        started, info = worker.start()
        if not started:
            self.events.put(("error", info))
            return
        self.events.put(("status", f"Converting on {info.upper()}…"))
        ok, detail = worker.convert(pdf, out_dir, options, skip_ocr)
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
        unit = info.get("unit") or ("page" if total == "1" else "pages")
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

        # Decided per file as the batch runs: scans in the queue keep their OCR.
        self.skip_ocr = tk.BooleanVar(value=False)
        ttk.Checkbutton(self, text=BATCH_SKIP_OCR_LABEL, variable=self.skip_ocr).grid(
            row=4, column=0, sticky="w", pady=(12, 0)
        )

        actions = ttk.Frame(self)
        actions.grid(row=5, column=0, pady=(18, 8))
        self.btn = ttk.Button(actions, text="Convert All", command=self.convert)
        self.btn.grid(row=0, column=0, padx=(0, 8))
        self.stop_btn = ttk.Button(actions, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.grid(row=0, column=1)

        ttk.Label(self, text="Overall", foreground=MUTED).grid(row=6, column=0, sticky="w")
        self.overall = ttk.Progressbar(self, length=420, mode="determinate", maximum=100)
        self.overall.grid(row=7, column=0, sticky="ew", pady=(2, 8))

        ttk.Label(self, text="Current file", foreground=MUTED).grid(row=8, column=0, sticky="w")
        self.progress = ttk.Progressbar(self, length=420, mode="determinate", maximum=100)
        self.progress.grid(row=9, column=0, sticky="ew", pady=(2, 6))

        self.status = tk.StringVar(value="No files queued")
        ttk.Label(self, textvariable=self.status, foreground=MUTED).grid(row=10, column=0, sticky="w")

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
        options = self.app.settings_tab.apply()
        if options is None:
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
        threading.Thread(
            target=self._run,
            args=(list(self.files), self.out_path.get(), options, self.skip_ocr.get()),
            daemon=True,
        ).start()
        self.after(100, self._drain_events)

    def stop(self) -> None:
        """Cancel the batch. This kills the worker, so the models reload on the
        next run — an acceptable trade for interrupting the in-flight file."""
        self.cancel.set()
        self.status.set("Stopping…")
        self.stop_btn.config(state="disabled")
        self.app.worker.stop()

    def _run(
        self,
        files: list[Path],
        out_dir: str,
        options: ImageExportSettings,
        skip_ocr_where_possible: bool = False,
    ) -> None:
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
                skip_ocr = skip_ocr_where_possible and can_skip_ocr(str(pdf))
                ok, detail = worker.convert(str(pdf), out_dir, options, skip_ocr)
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
        unit = info.get("unit") or ("page" if total == "1" else "pages")
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
    def __init__(self, settings_path: Path = SETTINGS_FILE):
        super().__init__()
        self.title("PDF → Markdown with Marker")
        self.minsize(720, 640)
        self.busy = False  # one conversion at a time — the worker is single-threaded
        self.worker = MarkerWorker()
        self.settings_path = settings_path
        self.image_settings, load_warning = load_image_settings(settings_path)
        self.export_summary = tk.StringVar(value=self.image_settings.summary())

        notebook = ttk.Notebook(self)
        self.notebook = notebook
        notebook.pack(fill="both", expand=True, padx=12, pady=(12, 0))
        notebook.add(SingleTab(notebook, self), text="Single")
        notebook.add(BatchTab(notebook, self), text="Batch")
        self.settings_tab = SettingsTab(notebook, self, load_warning)
        notebook.add(self.settings_tab, text="Settings")

        ttk.Label(self, textvariable=self.export_summary).pack(anchor="w", padx=16, pady=(8, 0))
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
