"""Textual TUI for the Muse Waffle text-to-sheet-music pipeline."""

from __future__ import annotations
import asyncio
import shutil
import threading
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button, Footer, Input, Label, Log, Static
)
from textual.reactive import reactive

from src.dataset import download_dataset, dataset_status, load_cached_paths
from src.model import build_model, load_model, model_status, DEFAULT_CONFIG, WEIGHTS_FILE, CONFIG_FILE
from src.trainer import train
from src.generator import generate_and_export, show_text_score


CSS = """
Screen {
    background: $surface;
}
#main-layout {
    height: 1fr;
    margin: 1 2;
}
#sidebar {
    width: 26;
    border: solid $primary;
    padding: 1 1;
}
#sidebar-title {
    text-style: bold;
    color: $primary;
    content-align: center middle;
    width: 100%;
    margin-bottom: 1;
}
#log-panel {
    border: solid $primary;
    padding: 0 1;
    width: 1fr;
    margin-left: 1;
}
#log-title {
    text-style: bold;
    color: $primary;
    margin-bottom: 0;
}
Button {
    width: 100%;
    margin-bottom: 1;
    height: 1;
    border: none;
    padding: 0 0;
    content-align: left middle;
}
.setting-row {
    height: 1;
    margin-bottom: 1;
}
.setting-label {
    width: 9;
    color: $text-muted;
    content-align: left middle;
}
.setting-input {
    width: 1fr;
    height: 1;
    border: none;
    padding: 0 0;
    background: $boost;
}
#status-bar {
    height: 1;
    margin: 0 2;
    color: $text-muted;
}
#input-area {
    height: 4;
    margin: 0 2 1 2;
    border: solid $accent;
    padding: 0 1;
}
#input-label {
    color: $accent;
    width: auto;
    margin-right: 1;
    content-align: left middle;
    height: 100%;
}
#prompt-input {
    width: 1fr;
}
.train-config {
    color: $text-muted;
    margin-bottom: 0;
    width: 100%;
}
"""


class MuseWaffleApp(App):
    CSS = CSS
    TITLE = "Muse Waffle v3"
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear_log", "Clear log"),
        Binding("f1", "download", "Dataset"),
        Binding("f2", "build_model", "Build Model"),
        Binding("f3", "do_train", "Train"),
        Binding("f4", "do_test", "Generate"),
        Binding("f5", "save_model", "Save Model"),
        Binding("f6", "load_model_action", "Load Model"),
        Binding("f7", "load_checkpoint", "Load Checkpoint"),
    ]

    _busy: reactive[bool] = reactive(False)
    _stop_flag: list[bool]

    def __init__(self):
        super().__init__()
        self._stop_flag = [False]

    # ── Layout ──────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        num_paths = str(len(load_cached_paths()))
        with Horizontal(id="main-layout"):
            with Vertical(id="sidebar"):
                yield Label("⚙  Commands", id="sidebar-title")
                yield Button("F1  Download Dataset",  id="btn-dataset", variant="primary")
                yield Button("F2  Build Model",        id="btn-model",   variant="primary")
                yield Button("F3  Train",              id="btn-train",   variant="success")
                yield Button("F4  Generate / Test",    id="btn-test",    variant="warning")
                yield Button("⏹  Stop",               id="btn-stop",    variant="error")
                yield Button("F5  Save Model",         id="btn-save",    variant="primary")
                yield Button("F6  Load Snapshot",       id="btn-load",    variant="primary")
                yield Button("F7  Load Checkpoint",    id="btn-ckpt",    variant="primary")
                yield Button("Clr Clear Log",          id="btn-clear",   variant="default")
                yield Static("── Settings ──────────", classes="train-config")
                with Horizontal(classes="setting-row"):
                    yield Label("Training Scores",  classes="setting-label")
                    yield Input(num_paths,  id="inp-training-scores", classes="setting-input")
                with Horizontal(classes="setting-row"):
                    yield Label("Epochs",  classes="setting-label")
                    yield Input("10",  id="inp-epochs", classes="setting-input")
                with Horizontal(classes="setting-row"):
                    yield Label("Batch",   classes="setting-label")
                    yield Input("4",   id="inp-batch",  classes="setting-input")
                with Horizontal(classes="setting-row"):
                    yield Label("LR",      classes="setting-label")
                    yield Input("1e-3", id="inp-lr",   classes="setting-input")
                with Horizontal(classes="setting-row"):
                    yield Label("Temp",    classes="setting-label")
                    yield Input("0.9", id="inp-temp",  classes="setting-input")
                with Horizontal(classes="setting-row"):
                    yield Label("TopK",    classes="setting-label")
                    yield Input("50",  id="inp-topk",  classes="setting-input")
                with Horizontal(classes="setting-row"):
                    yield Label("Device",  classes="setting-label")
                    yield Input("cpu", id="inp-device", classes="setting-input")

            with Vertical(id="log-panel"):
                yield Label("Output Log", id="log-title")
                yield Log(id="app-log", highlight=True)

        with Horizontal(id="input-area"):
            yield Label("Prompt ▶", id="input-label")
            yield Input(
                placeholder="e.g.  slow romantic piano piece in C major",
                id="prompt-input",
            )

        yield Static("Ready  |  F1 Download  F2 Build  F3 Train  F4 Generate  F5 Save  F6 Load  Ctrl+C Quit",
                     id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        self._log("Welcome to Muse Waffle v3!")
        self._log("Dataset  : " + dataset_status())
        self._log("Model    : " + model_status())
        self._log("Press F1 to download the OpenScore dataset to get started.")

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        log_widget = self.query_one("#app-log", Log)
        log_widget.write_line(msg)

    def _set_status(self, msg: str):
        self.query_one("#status-bar", Static).update(msg)

    def _run_in_thread(self, fn, *args):
        self._stop_flag[0] = False
        self._busy = True
        self._set_status("⏳ Working…  (press ⏹ Stop to cancel)")

        def wrapper():
            try:
                fn(*args)
            except Exception as e:
                self.call_from_thread(self._log, f"[red]ERROR: {e}[/red]")
            finally:
                self.call_from_thread(self._finish_work)

        threading.Thread(target=wrapper, daemon=True).start()

    def _finish_work(self):
        self._busy = False
        self._set_status("Ready  |  F1 Download  F2 Build  F3 Train  F4 Generate  F5 Save  F6 Load  Ctrl+C Quit")

    def _thread_log(self, msg: str):
        self.call_from_thread(self._log, msg)

    # ── Actions ─────────────────────────────────────────────────────────────

    def action_clear_log(self):
        self.query_one("#app-log", Log).clear()

    def action_download(self):
        if self._busy:
            self._log("Already running — stop current task first.")
            return
        self._run_in_thread(self._download_task)

    def action_build_model(self):
        if self._busy:
            self._log("Already running — stop current task first.")
            return
        self._run_in_thread(self._build_model_task)

    def action_do_train(self):
        if self._busy:
            self._log("Already running — stop current task first.")
            return
        try:
            epochs     = int(self.query_one("#inp-epochs", Input).value)
            batch_size = int(self.query_one("#inp-batch",  Input).value)
            lr         = float(self.query_one("#inp-lr",   Input).value)
        except ValueError as e:
            self._log(f"[red]Invalid setting: {e}[/red]")
            return
        device_pref = self.query_one("#inp-device", Input).value.strip().lower() or "cpu"
        self._run_in_thread(self._train_task, epochs, batch_size, lr, device_pref)

    def action_do_test(self):
        if self._busy:
            self._log("Already running — stop current task first.")
            return
        prompt = self.query_one("#prompt-input", Input).value.strip()
        if not prompt:
            self._log("[yellow]Enter a text prompt in the box below before generating.[/yellow]")
            return
        try:
            temperature = float(self.query_one("#inp-temp", Input).value)
            top_k       = int(self.query_one("#inp-topk",   Input).value)
        except ValueError as e:
            self._log(f"[red]Invalid setting: {e}[/red]")
            return
        self._run_in_thread(self._generate_task, prompt, temperature, top_k)

    def action_save_model(self):
        name = self.query_one("#prompt-input", Input).value.strip().replace(" ", "_") or "snapshot"
        self._run_in_thread(self._save_model_task, name)

    def action_load_model_action(self):
        name = self.query_one("#prompt-input", Input).value.strip().replace(" ", "_") or "snapshot"
        self._run_in_thread(self._load_model_task, name)

    def action_load_checkpoint(self):
        if self._busy:
            self._log("Already running — stop current task first.")
            return
        self._run_in_thread(self._load_checkpoint_task)

    # ── Button → Action bridge ───────────────────────────────────────────────

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id
        if btn_id == "btn-dataset":
            self.action_download()
        elif btn_id == "btn-model":
            self.action_build_model()
        elif btn_id == "btn-train":
            self.action_do_train()
        elif btn_id == "btn-test":
            self.action_do_test()
        elif btn_id == "btn-stop":
            self._stop_flag[0] = True
            self._log("[yellow]Stop requested…[/yellow]")
        elif btn_id == "btn-save":
            self.action_save_model()
        elif btn_id == "btn-load":
            self.action_load_model_action()
        elif btn_id == "btn-ckpt":
            self.action_load_checkpoint()
        elif btn_id == "btn-clear":
            self.action_clear_log()

    # ── Background tasks ─────────────────────────────────────────────────────

    def _download_task(self):
        self._thread_log("─" * 50)
        self._thread_log("Downloading OpenScore/Lieder dataset…")

        def progress(done, total):
            if total > 0:
                pct = int(100 * done / total)
                self.call_from_thread(
                    self._set_status, f"Downloading…  {done}/{total}  ({pct}%)"
                )

        num_paths = int(self.query_one("#inp-training-scores", Input).value)
        count = download_dataset(self._thread_log, progress, max_files=num_paths)
        self._thread_log(f"Done. {count} files cached.")

    def _build_model_task(self):
        self._thread_log("─" * 50)
        self._thread_log("Building model with default config…")
        model = build_model(DEFAULT_CONFIG)
        self._thread_log(f"Model created: {model.num_params():,} parameters.")
        self._thread_log(f"Config: {DEFAULT_CONFIG}")

    def _train_task(self, epochs: int, batch_size: int, lr: float, device_pref: str = "cpu"):
        import torch
        self._thread_log("─" * 50)
        self._thread_log(f"Starting training  epochs={epochs}  batch={batch_size}  lr={lr}")

        paths = load_cached_paths()
        if not paths:
            self._thread_log("[red]No dataset found. Run Download Dataset first.[/red]")
            return

        if device_pref == "cuda" and not torch.cuda.is_available():
            self._thread_log("[yellow]CUDA not available, falling back to cpu.[/yellow]")
            device_pref = "cpu"
        device = device_pref
        self._thread_log(f"Device: {device}")
        model, _ = load_model(device)

        def on_epoch(epoch: int, loss: float):
            self.call_from_thread(self._log, f"  Epoch {epoch}  avg_loss={loss:.4f}")
            self.call_from_thread(
                self._set_status, f"Training  epoch {epoch}/{epochs}  loss={loss:.4f}"
            )

        train(
            model, paths, epochs, batch_size, lr, device,
            self._thread_log, on_epoch, self._stop_flag,
        )

    def _save_model_task(self, name: str):
        self._thread_log("─" * 50)
        if not WEIGHTS_FILE.exists():
            self._thread_log("[red]No trained weights to save. Train first.[/red]")
            return
        snap_dir = Path("checkpoints/snapshots")
        snap_dir.mkdir(parents=True, exist_ok=True)
        dest_w = snap_dir / f"{name}.pt"
        dest_c = snap_dir / f"{name}.json"
        shutil.copy2(WEIGHTS_FILE, dest_w)
        shutil.copy2(CONFIG_FILE, dest_c)
        self._thread_log(f"Saved snapshot → {dest_w}")
        self._thread_log(f"Saved config   → {dest_c}")

    def _load_checkpoint_task(self):
        self._thread_log("─" * 50)
        if not WEIGHTS_FILE.exists():
            self._thread_log("[red]No checkpoint found at checkpoints/model.pt — train first.[/red]")
            return
        size = WEIGHTS_FILE.stat().st_size // 1024
        self._thread_log(f"Loading last checkpoint ({size} KB)…")
        try:
            model, cfg = load_model("cpu")
            self._thread_log(f"Checkpoint loaded: {model.num_params():,} params")
            self._thread_log(f"Config: {cfg}")
            self._thread_log("Ready — use F4 to generate.")
        except Exception as e:
            self._thread_log(f"[red]Failed to load checkpoint: {e}[/red]")

    def _load_model_task(self, name: str):
        self._thread_log("─" * 50)
        snap_dir = Path("checkpoints/snapshots")
        src_w = snap_dir / f"{name}.pt"
        src_c = snap_dir / f"{name}.json"
        if not src_w.exists():
            available = [p.stem for p in snap_dir.glob("*.pt")] if snap_dir.exists() else []
            self._thread_log(f"[red]Snapshot '{name}' not found.[/red]")
            if available:
                self._thread_log(f"Available: {', '.join(available)}")
            else:
                self._thread_log("No snapshots saved yet. Use F5 to save one.")
            return
        shutil.copy2(src_w, WEIGHTS_FILE)
        shutil.copy2(src_c, CONFIG_FILE)
        self._thread_log(f"Loaded snapshot '{name}' → {WEIGHTS_FILE}")
        self._thread_log("Model ready. Use F4 to generate.")

    def _generate_task(self, prompt: str, temperature: float, top_k: int):
        self._thread_log("─" * 50)
        self._thread_log(f"Generating music for: '{prompt}'  temp={temperature}  topk={top_k}")
        paths = generate_and_export(
            prompt=prompt,
            max_new=2000,
            temperature=temperature,
            top_k=top_k,
            log=self._thread_log,
        )
        if paths["xml"]:
            self._thread_log("─ Score preview (music21) ─")
            show_text_score(paths["xml"], self._thread_log)
        self._thread_log(f"Files saved to [b]{Path('output').resolve()}[/b]")
