# ui_tk.py (fixed)
# - Start button no longer clipped (layout reflowed to two rows + column weights)
# - Engine actually plays on start (creates instruments + seeds tensions)

import asyncio
import threading
import tkinter as tk
from tkinter import ttk, messagebox

from orchestrator import GlobalConfig, PortRegistry, Orchestrator, md


# --------------------------- Async engine host ---------------------------
class AsyncEngine:
    """Runs orchestrator in a background asyncio loop. Thread-safe setters.
    """
    def __init__(self):
        self.cfg = GlobalConfig(
            addr="127.0.0.1",
            bpm=120,
            beats_per_bar=4,
            channels={
                "drums": 10,
                "harm.voice1": 1,
                "harm.voice2": 2,
                "harm.voice3": 3,
                "lead": 4,
            },
            registers={
                "harm.voice1": md.note_to_int("C3"),
                "harm.voice2": md.note_to_int("E4"),
                "harm.voice3": md.note_to_int("G5"),
                "lead": md.note_to_int("C5"),
            },
        )
        self.ports = PortRegistry({
            "drums": 9000,
            "harm.voice1": 9001,
            "harm.voice2": 9002,
            "harm.voice3": 9003,
            "lead": 9004,
        })

        self.loop: asyncio.AbstractEventLoop | None = None
        self.thread: threading.Thread | None = None
        self.orch: Orchestrator | None = None
        self._run_future = None

    # ---- lifecycle ----
    def start(self):
        if self.loop is not None:
            return
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._thread_main, daemon=True)
        self.thread.start()

        # Construct orchestrator and instruments on the main thread.
        self.orch = Orchestrator(self.cfg, self.ports)
        self.orch.add_middleware(lambda msg: msg)  # no-op for debugging hooks
        # IMPORTANT: create instruments so something actually plays
        self.orch.add_percussion("drums")
        self.orch.add_harmonic_group("harm", ["harm.voice1", "harm.voice2", "harm.voice3"])
        self.orch.add_melodic("lead")
        # Seed tensions so harmony chooses chords immediately
        self.orch.set_tension("zone1", 0.2)
        self.orch.set_tension("zone2", 0.4)
        self.orch.set_tension("zone3", 0.7)

        # Run orchestrator in the background loop
        self._run_future = asyncio.run_coroutine_threadsafe(self.orch.run(), self.loop)

    def _thread_main(self):
        assert self.loop is not None
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def stop(self):
        if not self.loop:
            return
        try:
            if self.orch:
                self.orch.stop()  # stops the clock
            if self._run_future:
                try:
                    self._run_future.result(timeout=0.1)
                except Exception:
                    pass
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            if self.thread:
                self.thread.join(timeout=1.0)
            self.loop = None
            self.thread = None
            self.orch = None
            self._run_future = None

    # ---- thread-safe control helpers ----
    def set_bpm(self, bpm: float):
        if self.orch:
            self.orch.set_bpm(float(bpm))

    def set_beats_per_bar(self, bpb: int):
        if self.orch:
            self.orch.cfg.beats_per_bar = int(bpb)

    def set_key_root(self, key_root: str):
        if not self.orch:
            return
        self.orch.cfg.key_root = key_root
        # update harmony engines
        for inst in self.orch._harm_groups.values():
            from melody import HarmonyGenerator
            inst.hg = HarmonyGenerator(key=key_root)

    def set_tension(self, zone: str, value: float):
        if self.orch:
            self.orch.set_tension(zone, float(value))

    def queue_chord(self, symbol: str):
        if self.orch:
            self.orch.queue_next_chord("harm", symbol)

    def trigger_fill(self, preset: str, beats: int):
        if self.orch:
            drums = self.orch._percussion.get("drums")
            if drums:
                drums.queue_fill_beats(int(beats), preset=preset)

    def play_lead(self, note_name: str, vel: float, dur: float):
        if not self.orch:
            return
        lead = self.orch._melodic.get("lead")
        if not lead:
            return
        try:
            if isinstance(note_name, str) and any(c.isalpha() for c in note_name):
                note = md.note_to_int(note_name.strip())
            else:
                note = int(note_name)
        except Exception:
            return
        lead.play(note, float(vel), float(dur))


# --------------------------- Tkinter UI ---------------------------
class OrchestratorUI(tk.Tk):
    KEYS = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

    def __init__(self):
        super().__init__()
        self.title("Realtime Orchestrator Control")
        # Widen a bit to avoid clipping; allow resize
        self.geometry("840x560")
        self.minsize(720, 520)

        # use Tk scaling for HiDPI friendliness
        try:
            self.tk.call('tk', 'scaling', 1.2)
        except Exception:
            pass

        self.engine = AsyncEngine()

        self._build_controls()

        # Start engine immediately AND provide Start button
        self._on_start()
        # also seed tensions (UI sliders will update too)
        for z, v in [("zone1", 0.2), ("zone2", 0.4), ("zone3", 0.7)]:
            self.engine.set_tension(z, v)
        self._refresh_status("Engine started.")

        # Heartbeat keeps UI responsive
        self.after(250, self._heartbeat)

    def _build_controls(self):
        pad = dict(padx=10, pady=8)

        # --- Transport / Global ---
        frm_global = ttk.LabelFrame(self, text="Transport / Global")
        frm_global.pack(fill="x", **pad)

        # row 0: BPM, Beats/Bar, Key
        ttk.Label(frm_global, text="BPM").grid(row=0, column=0, sticky="w")
        self.var_bpm = tk.DoubleVar(value=120.0)
        bpm_spin = ttk.Spinbox(frm_global, from_=40, to=240, increment=1, textvariable=self.var_bpm, width=7)
        bpm_spin.grid(row=0, column=1, sticky="w")
        ttk.Button(frm_global, text="Set", command=self._on_set_bpm).grid(row=0, column=2, sticky="w", padx=6)

        ttk.Label(frm_global, text="Beats/Bar").grid(row=0, column=3, sticky="w")
        self.var_bpb = tk.IntVar(value=4)
        bpb_spin = ttk.Spinbox(frm_global, from_=1, to=12, increment=1, textvariable=self.var_bpb, width=7)
        bpb_spin.grid(row=0, column=4, sticky="w")
        ttk.Button(frm_global, text="Set", command=self._on_set_bpb).grid(row=0, column=5, sticky="w", padx=6)

        ttk.Label(frm_global, text="Key").grid(row=0, column=6, sticky="w")
        self.var_key = tk.StringVar(value="C")
        key_combo = ttk.Combobox(frm_global, values=self.KEYS, textvariable=self.var_key, width=5, state="readonly")
        key_combo.grid(row=0, column=7, sticky="w")
        ttk.Button(frm_global, text="Set", command=self._on_set_key).grid(row=0, column=8, sticky="w", padx=6)

        # Make columns 1,4,7 expand a bit
        for c in (1,4,7):
            frm_global.grid_columnconfigure(c, weight=1)

        # row 1: Start/Stop aligned right, never clipped
        btns = ttk.Frame(frm_global)
        btns.grid(row=1, column=0, columnspan=9, sticky="e", pady=(6,0))
        ttk.Button(btns, text="Start Engine", command=self._on_start).pack(side="right", padx=6)
        ttk.Button(btns, text="Stop Engine", command=self._on_stop).pack(side="right")

        # --- Tensions ---
        frm_t = ttk.LabelFrame(self, text="Zone Tensions (0..1)")
        frm_t.pack(fill="x", **pad)

        self.sliders = {}
        for i, zone in enumerate(("zone1","zone2","zone3")):
            ttk.Label(frm_t, text=zone).grid(row=0, column=2*i, sticky="w")
            var = tk.DoubleVar(value={"zone1":0.2, "zone2":0.4, "zone3":0.7}[zone])
            s = ttk.Scale(frm_t, from_=0.0, to=1.0, orient="horizontal",
                          variable=var, command=lambda _v, z=zone, sv=var: self._on_tension(z, sv))
            s.grid(row=0, column=2*i+1, sticky="ew")
            frm_t.grid_columnconfigure(2*i+1, weight=1)
            self.sliders[zone] = var

        # --- Drums / Fills ---
        frm_d = ttk.LabelFrame(self, text="Drums")
        frm_d.pack(fill="x", **pad)

        ttk.Label(frm_d, text="Fill preset").grid(row=0, column=0, sticky="w")
        self.var_fill_preset = tk.StringVar(value="snare")
        ttk.Combobox(frm_d, values=["snare","toms","hats"], textvariable=self.var_fill_preset,
                     state="readonly", width=10).grid(row=0, column=1, sticky="w")

        ttk.Label(frm_d, text="Fill beats").grid(row=0, column=2, sticky="w", padx=(12,0))
        self.var_fill_beats = tk.IntVar(value=3)
        ttk.Spinbox(frm_d, from_=1, to=32, textvariable=self.var_fill_beats, width=7).grid(row=0, column=3, sticky="w")

        ttk.Button(frm_d, text="Trigger Fill", command=self._on_fill).grid(row=0, column=4, sticky="w", padx=10)

        # --- Harmony / chord override ---
        frm_h = ttk.LabelFrame(self, text="Harmony")
        frm_h.pack(fill="x", **pad)

        ttk.Label(frm_h, text="Override (e.g., C_M, F_M7)").grid(row=0, column=0, sticky="w")
        self.var_chord = tk.StringVar(value="C_M")
        ttk.Entry(frm_h, textvariable=self.var_chord, width=12).grid(row=0, column=1, sticky="w")
        ttk.Button(frm_h, text="Queue for Next Bar", command=self._on_chord_override).grid(row=0, column=2, sticky="w", padx=8)

        # --- Melodic test ---
        frm_m = ttk.LabelFrame(self, text="Melodic (Lead) Test Note")
        frm_m.pack(fill="x", **pad)

        ttk.Label(frm_m, text="Note").grid(row=0, column=0, sticky="w")
        self.var_note = tk.StringVar(value="A4")
        ttk.Entry(frm_m, textvariable=self.var_note, width=8).grid(row=0, column=1, sticky="w")

        ttk.Label(frm_m, text="Vel").grid(row=0, column=2, sticky="w")
        self.var_vel = tk.DoubleVar(value=0.8)
        ttk.Spinbox(frm_m, from_=0.0, to=1.0, increment=0.05, textvariable=self.var_vel, width=7).grid(row=0, column=3, sticky="w")

        ttk.Label(frm_m, text="Dur(s)").grid(row=0, column=4, sticky="w")
        self.var_dur = tk.DoubleVar(value=0.12)
        ttk.Spinbox(frm_m, from_=0.02, to=2.0, increment=0.02, textvariable=self.var_dur, width=7).grid(row=0, column=5, sticky="w")

        ttk.Button(frm_m, text="Play", command=self._on_play_lead).grid(row=0, column=6, sticky="w", padx=8)

        # --- Status / Log ---
        frm_log = ttk.LabelFrame(self, text="Status")
        frm_log.pack(fill="both", expand=True, **pad)
        self.txt_status = tk.Text(frm_log, height=8)
        self.txt_status.pack(fill="both", expand=True)

        # --- Quit ---
        frm_q = ttk.Frame(self)
        frm_q.pack(fill="x", **pad)
        ttk.Button(frm_q, text="Quit", command=self._on_quit).pack(side="right")

    # --------------------------- UI callbacks ---------------------------
    def _on_set_bpm(self):
        bpm = float(self.var_bpm.get())
        self.engine.set_bpm(bpm)
        self._refresh_status(f"BPM -> {bpm:.0f}")

    def _on_set_bpb(self):
        bpb = int(self.var_bpb.get())
        bpb = max(1, bpb)
        self.engine.set_beats_per_bar(bpb)
        self._refresh_status(f"Beats/Bar -> {bpb}")

    def _on_set_key(self):
        key = self.var_key.get().strip()
        if key not in self.KEYS:
            messagebox.showerror("Key", f"Unsupported key: {key}")
            return
        self.engine.set_key_root(key)
        self._refresh_status(f"Key root -> {key}")

    def _on_tension(self, zone: str, var: tk.DoubleVar):
        self.engine.set_tension(zone, float(var.get()))

    def _on_fill(self):
        self.engine.trigger_fill(self.var_fill_preset.get(), int(self.var_fill_beats.get()))
        self._refresh_status(
            f"Drum fill: preset={self.var_fill_preset.get()}, beats={int(self.var_fill_beats.get())}"
        )

    def _on_chord_override(self):
        symbol = self.var_chord.get().strip()
        if "_" not in symbol:
            messagebox.showerror("Chord", "Chord must look like 'C_M' or 'F_M7'.")
            return
        self.engine.queue_chord(symbol)
        self._refresh_status(f"Queued chord override for next bar: {symbol}")

    def _on_play_lead(self):
        self.engine.play_lead(self.var_note.get(), float(self.var_vel.get()), float(self.var_dur.get()))
        self._refresh_status(
            f"Lead: {self.var_note.get()} vel={float(self.var_vel.get()):.2f} dur={float(self.var_dur.get()):.2f}s"
        )

    def _on_start(self):
        self.engine.start()
        self._refresh_status("Engine started.")

    def _on_stop(self):
        self.engine.stop()
        self._refresh_status("Engine stopped.")

    def _on_quit(self):
        try:
            self.engine.stop()
        finally:
            self.destroy()

    def _heartbeat(self):
        # placeholder: could show beat indicator
        self.after(250, self._heartbeat)

    def _refresh_status(self, msg: str):
        self.txt_status.insert("end", msg + "\n")
        self.txt_status.see("end")


if __name__ == "__main__":
    app = OrchestratorUI()
    app.mainloop()
