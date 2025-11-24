# orchestrator.py
import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterable, Union, Callable

# ---- External deps you already have ----
import common.midi as md
from melody import HarmonyGenerator
from common.osc import OSCManager

# Re-export md so you can `from orchestrator import md` in main.py
__all__ = ["GlobalConfig", "PortRegistry", "Orchestrator", "md"]

Scalar = Union[int, float, str, bytes, bool]
Arg = Union[Scalar, Iterable[Scalar]]

# =========================
#         CORE
# =========================

@dataclass
class GlobalConfig:
    addr: str = "127.0.0.1"
    bpm: float = 120.0
    beats_per_bar: int = 4
    sub_per_beat: int = 4             # 16ths
    key_root: str = "C"
    scale_name: str = "major"
    channels: Dict[str, int] = field(default_factory=dict)   # instrument_id -> MIDI channel
    registers: Dict[str, int] = field(default_factory=dict)  # instrument_id -> target register (MIDI note)

@dataclass
class PortRegistry:
    ports: Dict[str, int]

    def add(self, instrument_id: str, port: int):
        self.ports[instrument_id] = port

    def get(self, instrument_id: str) -> int:
        return self.ports[instrument_id]

# ------------------------- Transport -------------------------

@dataclass
class OscMessage:
    port: int
    address: str
    args: List[Arg]

Middleware = Callable[[OscMessage], Optional[OscMessage]]

class Transport:
    """
    OSC out with middleware. Provides:
    - send_now(): synchronous (grid-critical)
    - send(): async (non-critical)
    """
    def __init__(self, addr: str, ports: Iterable[int]):
        # Materialize to a list before splatting to avoid iterator pitfalls.
        port_list = list(ports)
        self._osc = OSCManager(addr, *port_list)
        self._middleware: List[Middleware] = []
        self._lock = asyncio.Lock()

    def add_middleware(self, fn: Middleware):
        self._middleware.append(fn)

    def _apply_mw(self, msg: OscMessage) -> Optional[OscMessage]:
        out = msg
        for mw in self._middleware:
            out = mw(out)
            if out is None:
                return None
        return out

    def send_now(self, msg: OscMessage):
        out = self._apply_mw(msg)
        if out is not None:
            self._osc(out.port, out.address, out.args)

    async def send(self, msg: OscMessage):
        async with self._lock:
            out = self._apply_mw(msg)
            if out is not None:
                self._osc(out.port, out.address, out.args)

# ------------------------- Tempo Clock -------------------------

class TempoClock:
    """Beat-accurate async ticker yielding (beat, sub) with live BPM updates."""
    def __init__(self, cfg: GlobalConfig):
        self.cfg = cfg
        self._running = True
        self.t0 = time.perf_counter()
        self.beat0 = 0.0

    def stop(self):
        self._running = False

    async def ticker(self):
        beat, sub = 0, 0
        while self._running:
            spb = 60.0 / max(20.0, min(240.0, self.cfg.bpm))
            target = self.t0 + (beat + sub / self.cfg.sub_per_beat - self.beat0) * spb
            delay = max(0.0, target - time.perf_counter())
            if delay > 0:
                await asyncio.sleep(delay)
            yield beat, sub
            sub += 1
            if sub >= self.cfg.sub_per_beat:
                sub = 0
                beat += 1

# ------------------------- Scheduler -------------------------

class Scheduler:
    """
    Phased scheduler:
    - on_downbeat: BAR downbeat only (sub==0 and beat%beats_per_bar==0) — commit point
    - on_tick: every subdivision
    - on_bar: informational hook at bar start (after downbeat commit)
    """
    def __init__(self, clock: TempoClock, cfg: GlobalConfig):
        self.clock, self.cfg = clock, cfg
        self._downbeat_handlers: List[Callable[[int], None]] = []
        self._tick_handlers: List[Callable[[int, int], None]] = []
        self._bar_handlers: List[Callable[[int], None]] = []

    def on_downbeat(self, fn: Callable[[int], None]): self._downbeat_handlers.append(fn)
    def on_tick(self, fn: Callable[[int, int], None]): self._tick_handlers.append(fn)
    def on_bar(self, fn: Callable[[int], None]): self._bar_handlers.append(fn)

    async def run(self):
        async for beat, sub in self.clock.ticker():
            bpb = self.cfg.beats_per_bar

            # BAR DOWNBEAT = commit point
            if sub == 0 and (beat % bpb == 0):
                bar_idx = beat // bpb
                for fn in list(self._downbeat_handlers):
                    fn(bar_idx)
                for fn in list(self._bar_handlers):
                    fn(bar_idx)

            # Subdivision phase
            for fn in list(self._tick_handlers):
                fn(beat, sub)

# =========================
#       INSTRUMENTS
# =========================

class Instrument:
    def __init__(self, instrument_id: str, cfg: GlobalConfig, ports: PortRegistry, tx: Transport):
        self.id, self.cfg, self.ports, self.tx = instrument_id, cfg, ports, tx

    def channel(self) -> int:
        return self.cfg.channels.get(self.id, 1)

    def register_center(self) -> int:
        return self.cfg.registers.get(self.id, md.note_to_int("C4"))

    def note_now(self, n: int, vel: float, dur: float):
        self.tx.send_now(OscMessage(self.ports.get(self.id), "/note", [int(n), float(vel), float(dur), int(self.channel())]))

    async def note_async(self, n: int, vel: float, dur: float):
        await self.tx.send(OscMessage(self.ports.get(self.id), "/note", [int(n), float(vel), float(dur), int(self.channel())]))

# ---------- Percussive ----------

from dataclasses import dataclass as _dc  # avoid name clash above

@_dc
class DrumSpec:
    mapping: Dict[str, int] = field(default_factory=lambda: {
        "kick":36, "snare":38, "chh":44, "ohh":46, "clap":39, "ride":51, "tom":45
    })
    base_vel: Dict[str, float] = field(default_factory=lambda: {
        "kick":0.85, "snare":0.8, "chh":0.6, "ohh":0.7, "clap":0.8, "ride":0.9, "tom":0.8
    })

class PercussiveInstrument(Instrument):
    """
    16-step default pattern in 4/4. Supports:
    - set_pattern(steps)
    - queue_fill_beats(fill_len_beats, preset='snare'|'toms'|'hats' or steps=list)
      Fill starts exactly `fill_len_beats` before next bar, ends on the downbeat.
    """
    def __init__(self, instrument_id: str, cfg: GlobalConfig, ports: PortRegistry, tx: Transport, spec: Optional[DrumSpec]=None):
        super().__init__(instrument_id, cfg, ports, tx)
        self.spec = spec or DrumSpec()
        self.pattern: List[Dict[str,int]] = [
            {"kick":1, "chh":1}, {}, {}, {},
            {"chh":1}, {}, {}, {},
            {"snare":1, "chh":1}, {}, {}, {},
            {"chh":1}, {}, {}, {},
            {"kick":1, "chh":1}, {}, {}, {},
            {"kick":1, "chh":1}, {}, {}, {},
            {"snare":1, "chh":1}, {}, {}, {},
            {"chh":1}, {}, {}, {},
            {"kick":1, "chh":1}, {}, {}, {},
            {"chh":1}, {}, {}, {},
            {"snare":1, "chh":1}, {}, {}, {},
            {"chh":1}, {}, {}, {},
            {"kick":1, "chh":1}, {}, {}, {},
            {"kick":1, "chh":1}, {}, {}, {},
            {"snare":1, "chh":1}, {}, {}, {},
            {"ohh":1}, {}, {}, {},
        ]
        self._pending_fill_beats: Optional[int] = None
        self._fill_steps: Optional[List[Dict[str,int]]] = None
        self._active_fill: Optional[List[Dict[str,int]]] = None

    def set_pattern(self, steps: List[Dict[str,int]]):
        self.pattern = steps

    def _preset_fill(self, beats: int, preset: str) -> List[Dict[str,int]]:
        steps = []
        tmp = 0
        for i in range(max(1, beats)):
            if preset == "toms":
                if tmp % 2 == 0:
                    steps.extend([{"tom":1}, {}])
                    steps.extend([{"tom":1}, {}])
                else:
                    steps.extend([{}, {}, {"tom": 1}, {}])
                tmp += 1
            elif preset == "hats":
                steps.extend([{"chh":1}, {"chh":1}, {}, {"chh":1}])
            elif preset == "snare":
                for j in range(2):
                    if tmp % 3 == 2:
                        steps.extend([{"kick": 1}, {}])
                    else:
                        steps.extend([{"snare": 1}, {}])
                    tmp += 1

        steps.append({"ride":1, "kick": "1"})
        return steps

    def queue_fill_beats(self, fill_len_beats: int, preset: str = "snare", steps: Optional[List[Dict[str,int]]] = None):
        self._pending_fill_beats = max(1, fill_len_beats)
        self._fill_steps = steps if steps is not None else self._preset_fill(self._pending_fill_beats, preset)

    def on_tick(self, beat: int, sub: int):
        # Start fill X beats before next bar (check at quarter notes to reduce branching).
        if sub == 0 and self._pending_fill_beats:
            bpb = self.cfg.beats_per_bar
            beats_to_bar = (bpb - (beat % bpb)) % bpb
            if beats_to_bar == self._pending_fill_beats:
                self._active_fill = list(self._fill_steps or [])
                self._pending_fill_beats = None

        # Fire pattern on every subdivision (16ths)
        step16 = (beat * self.cfg.sub_per_beat + sub) % len(self.pattern)

        # Fill overrides pattern while active
        if self._active_fill:
            step_dict = self._active_fill.pop(0)
            if not self._active_fill:
                self._active_fill = None
        else:
            step_dict = self.pattern[step16]

        # Emit hits immediately (tight)
        ch = self.channel()
        port = self.ports.get(self.id)
        for name, on in step_dict.items():
            if not on:
                continue
            nn = self.spec.mapping.get(name, 36)
            v  = min(1.0, self.spec.base_vel.get(name, 0.8))
            self.tx.send_now(OscMessage(port, "/note", [nn, v, 0.05, ch]))

    def on_downbeat(self, bar_idx: int):
        # No-op; per-beat/per-sub handling is in on_tick
        pass

# ---------- Harmonic ----------

class HarmonicInstrument(Instrument):
    """
    N voices -> N instrument_ids/ports (e.g., 'harm.voice1', 'harm.voice2', 'harm.voice3').
    Default chord is selected from current max tension; can queue an override for next downbeat.
    """
    def __init__(self, instrument_id: str, cfg: GlobalConfig, ports: PortRegistry, tx: Transport,
                 voice_ids: List[str], key_root: str):
        super().__init__(instrument_id, cfg, ports, tx)
        self.voice_ids = voice_ids
        self.hg = HarmonyGenerator(key=key_root)
        self._prev: Optional[str] = None
        self._queued_next: Optional[str] = None
        self._current_symbol: str = f"{cfg.key_root}_M"
        self._current_notes: List[int] = []

    def queue_next_chord(self, symbol: str):
        self._queued_next = symbol

    def _select_symbol(self, max_tension: float) -> str:
        if self._queued_next:
            sym = self._queued_next
            self._queued_next = None
            self._prev = sym
            return sym
        if max_tension > 0:
            sym = self.hg.select_chord_by_tension(
                current_tension=max_tension,
                previous_chord=self._prev,
                lambda_balance=1.0,
                k=1
            )
        else:
            sym = f"{self.cfg.key_root}_M"
        self._prev = sym
        return sym

    @staticmethod
    def chord_to_pitches(symbol: str) -> List[int]:
        root_str, ctype = symbol.split('_')
        root = md.note_to_int(root_str + "0")
        return [root + i for i in md.HARMONIES_SHORT[ctype]]

    def _voice_spread(self, base: List[int]) -> List[int]:
        base = sorted(base)
        # choose degrees R,3,7 if available else R,3,5
        if len(base) >= 4:
            degrees = [base[0], base[1], base[3]]
        elif len(base) == 3:
            degrees = base
        else:
            degrees = base + base[:max(0, 3 - len(base))]

        def near(n0: int, center: int) -> int:
            cands = [n0 + 12 * k for k in range(-6, 7)]
            return min(cands, key=lambda n: abs(n - center))

        voiced: List[int] = []
        defaults = [md.note_to_int("C3"), md.note_to_int("C4"), md.note_to_int("C5")]
        for i, vid in enumerate(self.voice_ids[:3]):
            center = self.cfg.registers.get(vid, defaults[i])
            voiced.append(near(degrees[i], center))
        return voiced

    def on_downbeat(self, bar_idx: int, max_tension: float, zone_tensions: Optional[Dict[str, float]] = None):
        """Commit chord for this bar and send immediately (synchronous)."""
        symbol = self._select_symbol(max_tension)

        # base pitches from the chord symbol (untransposed)
        base = self.chord_to_pitches(symbol)

        # --- transpose base by global key offset (C -> key_root) ---
        try:
            key_off = md.note_to_int(self.cfg.key_root + "0") - md.note_to_int("C0")
        except Exception:
            key_off = 0

        print(key_off)
        base = [n + key_off for n in base]

        voiced = self._voice_spread(base)
        self._current_symbol, self._current_notes = symbol, voiced

        spb = 60.0 / max(20.0, min(240.0, self.cfg.bpm))
        dur = spb * self.cfg.beats_per_bar - 0.03

        # Map voice IDs to zone tensions for velocity calculation
        voice_zone_map = {
            "harm.voice1": "zone1",
            "harm.voice2": "zone2", 
            "harm.voice3": "zone3"
        }

        for vid, n in zip(self.voice_ids, voiced):
            ch = self.cfg.channels.get(vid, self.channel())
            
            # Calculate velocity based on zone tension
            base_velocity = 0.5  # Minimum velocity (increased from 0.3)
            max_velocity = 1.0   # Maximum velocity (increased from 0.9)
            
            if zone_tensions and vid in voice_zone_map:
                zone = voice_zone_map[vid]
                tension = zone_tensions.get(zone, 0.0)
                # Scale tension (0-1) to velocity range (base_velocity to max_velocity)
                velocity = base_velocity + (max_velocity - base_velocity) * tension
                print(f"Voice {vid} -> {zone}: tension={tension:.3f}, velocity={velocity:.3f}")
            else:
                # Fallback to default velocity if no zone mapping
                velocity = 0.7
                print(f"Voice {vid}: using default velocity={velocity:.3f}")
            
            self.tx.send_now(OscMessage(
                port=self.ports.get(vid),
                address="/note",
                args=[n, velocity, dur, ch]
            ))

# ---------- Melodic ----------

class MelodicInstrument(Instrument):
    """Manual note control: forward to OSC immediately (tight)."""
    def play(self, note: int, vel: float, dur: float):
        self.note_now(note, vel, dur)

# =========================
#       ORCHESTRATOR
# =========================

class Orchestrator:
    """
    High-level facade:
    - Downbeat = commit point (harmonic decisions read latest tension/overrides)
    - Dynamic instruments: percussive, harmonic (N voice ports), melodic
    - Realtime middleware for OSC edits/monitoring
    """
    def __init__(self, cfg: GlobalConfig, ports: PortRegistry):
        self.cfg, self.ports = cfg, ports
        # Make sure we pass a concrete list of ports to Transport
        self._tx = Transport(cfg.addr, list(ports.ports.values()) or [9000])
        self._clock = TempoClock(cfg)
        self._sched = Scheduler(self._clock, cfg)

        self._tension: Dict[str, float] = {}  # zone -> [0..1]
        self._percussion: Dict[str, PercussiveInstrument] = {}
        self._melodic: Dict[str, MelodicInstrument] = {}
        self._harm_groups: Dict[str, HarmonicInstrument] = {}

        # Wire phases
        self._sched.on_downbeat(self._on_downbeat_phase)
        self._sched.on_tick(self._on_tick_phase)

    # ---- Public controls ----
    def add_middleware(self, fn: Middleware): self._tx.add_middleware(fn)
    def set_bpm(self, bpm: float): self.cfg.bpm = float(bpm)
    def set_tension(self, zone: str, val: float): self._tension[zone] = max(0.0, min(1.0, float(val)))
    def max_tension(self) -> float: return max(self._tension.values(), default=0.0)

    # ---- Builders ----
    def add_percussion(self, instrument_id: str) -> PercussiveInstrument:
        inst = PercussiveInstrument(instrument_id, self.cfg, self.ports, self._tx)
        self._percussion[instrument_id] = inst
        return inst

    def add_harmonic_group(self, group_id: str, voice_ids: List[str], key_root: Optional[str] = None) -> HarmonicInstrument:
        inst = HarmonicInstrument(group_id, self.cfg, self.ports, self._tx, voice_ids, key_root or self.cfg.key_root)
        self._harm_groups[group_id] = inst
        return inst

    def add_melodic(self, instrument_id: str) -> MelodicInstrument:
        inst = MelodicInstrument(instrument_id, self.cfg, self.ports, self._tx)
        self._melodic[instrument_id] = inst
        return inst

    def queue_next_chord(self, group_id: str, symbol: str):
        self._harm_groups[group_id].queue_next_chord(symbol)

    # ---- Scheduler phases ----
    def _on_downbeat_phase(self, bar_idx: int):
        # 1) Harmonic commit (tight, synchronous send)
        max_t = self.max_tension()
        for inst in self._harm_groups.values():
            inst.on_downbeat(bar_idx, max_t, self._tension)
        # 2) Percussion may choose to do bar-start work (currently on_tick drives steps)
        for inst in self._percussion.values():
            inst.on_downbeat(bar_idx)

        try:
            root_note = md.note_to_int(f"{self.cfg.key_root}5")
        except Exception:
            root_note = md.note_to_int("C5")


        spb = 60.0 / max(20.0, min(240.0, self.cfg.bpm))
        dur = spb * self.cfg.beats_per_bar - 0.03

        for _port in (8000, 8001):
            self._tx.send_now(OscMessage(_port, "/note", [int(root_note), 0.6, float(dur), 1]))


    def _on_tick_phase(self, beat: int, sub: int):
        # Percussion steps per subdivision (tight send_now inside)
        for inst in self._percussion.values():
            inst.on_tick(beat, sub)

    # ---- Lifecycle ----
    async def run(self):
        await self._sched.run()

    def stop(self):
        self._clock.stop()
