import asyncio, time, random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterable, Any, Union
import common.midi as md
from melody import HarmonyGenerator
from common.osc import OSCManager

Scalar = Union[int, float, str, bytes, bool]
Arg = Union[Scalar, Iterable[Scalar]]

# class HarmonyAdapter:
#     """
#     Thin adapter around the prior HarmonyGenerator.
#     We only need a 'choose_next_chord' that respects zone tensions.
#     """
#     def __init__(self, key_root: str = "C"):
#         self.hg = HarmonyGenerator(key_root)
#
#     def choose_next_chord(self, zone_tension: Dict[int, float]) -> str:
#         """
#         Choose a chord biased by the max (or weighted) zone tension.
#         Returns chord name like 'C_M' compatible with your midi.py sets.
#         """
#         # You can swap this with your composite weighting strategy.
#         tvals = list(zone_tension.values())
#         tmax = max(tvals) if tvals else 0.0
#         # Delegate to HG’s tension-based selector; previous chord tracking internally
#         return self.hg.select_chord_by_tension(tmax)
#
#     def chord_tones(self, chord: str) -> List[int]:
#         return self.hg.chord_to_list(chord)


# ---- Global state ----

@dataclass
class State:
    addr: str = "127.0.0.1"
    ports: Dict[str, int] = field(default_factory=lambda: {
        "drums": 9000,
        "zone1": 9001,
        "zone2": 9002,
        "zone3": 9003,
    })
    channels: Dict[str, int] = field(default_factory=lambda: {
        "drums": 10,  # GM drums
        "zone1": 10,
        "zone2": 2,
        "zone3": 3,
    })
    bpm: float = 120.0
    beats_per_bar: int = 4
    chord_span_beats: int = 4  # change chord every N beats
    key_root: str = "C"        # 'C','D#', etc.
    scale_name: str = "major"
    tonic_center: int = md.note_to_int("C4")  # central register anchor for voicing
    zone_registers: Dict[str, int] = field(default_factory=lambda: {
        "zone1": md.note_to_int("C3"),
        "zone2": md.note_to_int("C4"),
        "zone3": md.note_to_int("C5"),
    })
    zone_tension: Dict[str, float] = field(default_factory=lambda: {
        "zone1": 0.0, "zone2": 0.0, "zone3": 0.0
    })
    drum_map: Dict[str, int] = field(default_factory=lambda: {
        "kick":36, "snare":38, "chh":44, "ohh":43, "clap":39, "ride":51, "tom":45
    })
    running: bool = True

# ---- Time / clock ----

class TempoClock:
    def __init__(self, state: State):
        self.state = state
        self.t0 = time.perf_counter()
        self.beat0 = 0.0

    async def ticker(self, sub_per_beat: int = 4):
        beat = 0
        sub = 0
        while self.state.running:
            spb = 60.0 / max(20.0, min(240.0, self.state.bpm))
            target_time = self.t0 + (beat + sub / sub_per_beat - self.beat0) * spb
            delay = max(0.0, target_time - time.perf_counter())
            if delay > 0:
                await asyncio.sleep(delay)
            yield beat, sub
            sub += 1
            if sub >= sub_per_beat:
                sub = 0
                beat += 1

# ---- Harmony adapter (wraps your HarmonyGenerator) ----

class HarmonyEngine:
    def __init__(self, state: State):
        self.state = state
        self.hg = HarmonyGenerator(key=state.key_root)
        self.prev_chord: Optional[str] = None  # e.g., 'C_M'

    def next_chord(self, max_tension: float) -> str:
        """Use your select_chord_by_tension; default to 'C_M' at low tension."""
        chord = self.hg.select_chord_by_tension(
            current_tension=max_tension,
            previous_chord=self.prev_chord,
            lambda_balance=1.0,
            k=4
        ) if max_tension > 0 else f"{self.state.key_root}_M"
        self.prev_chord = chord
        return chord

    @staticmethod
    def chord_to_midi(chord: str) -> List[int]:
        # chord like 'C_M', use md.HARMONIES_SHORT intervals
        root_str, ctype = chord.split('_')
        root = md.note_to_int(root_str + "0")
        return [root + i for i in md.HARMONIES_SHORT[ctype]]

# ---- Instruments ----

class DrumKit:
    def __init__(self, osc: OSCManager, state: State):
        self.osc, self.state = osc, state
        self.pattern = [
            {"kick":1, "chh":1},
            {"chh":1},
            {"snare":1, "chh":1},
            {"chh":1},
            {"kick":1, "chh":1},
            {"kick":1, "chh":1},
            {"snare": 1, "chh": 1},
            {"chh":1},
        ]
        self.fill_queue: Optional[List[Dict[str,int]]] = None

    def queue_fill(self, steps: Optional[List[Dict[str,int]]] = None):
        if steps is None:
            steps = [{"snare":1} for _ in range(15)] + [{"ride":1, "snare":1}]
        self.fill_queue = steps

    def step(self, step16: int):
        port = self.state.ports["drums"]
        ch = self.state.channels["drums"]
        pat = self.fill_queue.pop(0) if self.fill_queue else self.pattern[step16 % len(self.pattern)]
        if self.fill_queue == []:
            self.fill_queue = None
        for name, on in pat.items():
            if on:
                nn = self.state.drum_map.get(name, 36)
                # MOSC: /note [note, vel01, durSec, ch]
                self.osc(port, "/note", [nn, 0.9, 0.05, ch])

class MelodicVoice:
    def __init__(self, zone_key: str, osc: OSCManager, state: State):
        self.zone, self.osc, self.state = zone_key, osc, state
        self.last_note: Optional[int] = None
        self.hold_until: float = 0.0

    def on_chord_stab(self, note: int):
        port = self.state.ports[self.zone]
        ch   = self.state.channels[self.zone]
        self.osc(port, "/note", [note, 0.6, 0.12, ch])
        self.last_note = note

    def melodic_step(self, sub: int, chord_notes: List[int], scale_notes: List[int]):
        # fire on 8ths
        t = float(self.state.zone_tension.get(self.zone, 0.0))
        if t < 1.0: return

        if sub % 2: return
        density = 0.3 + 0.7 * t
        if random.random() > density: return
        now = time.perf_counter()
        if now < self.hold_until: return

        center = self.state.zone_registers[self.zone]
        step_choices = [-2,-1,0,1,2,3] if t < 0.5 else [-4,-2,-1,0,1,2,3,5]
        base = self.last_note if self.last_note is not None else center
        candidate = base + random.choice(step_choices)

        # quantize to scale
        target = min(scale_notes, key=lambda n: abs(n - candidate))
        # maybe snap near chord
        if random.random() < (0.6 + 0.3 * t):
            target = min([p + 12*k for p in chord_notes for k in range(-2,3)], key=lambda n: abs(n - target))

        port = self.state.ports[self.zone]
        ch   = self.state.channels[self.zone]
        vel  = 0.5 + 0.5 * t
        dur  = 0.10 if t < 0.5 else 0.08
        self.osc(port, "/note", [target, vel, dur, ch])
        self.last_note = target
        self.hold_until = now + dur * (1.1 if t < 0.5 else 0.9)

# ---- Conductor ----

class Conductor:
    def __init__(self, state: State):
        self.state = state
        # Set up OSC across all ports (they can be on different MOSC instances)
        self.osc = OSCManager(state.addr, *state.ports.values())
        self.clock = TempoClock(state)
        self.harmony = HarmonyEngine(state)
        # Precompute scale notes around tonic center
        tonic = md.note_to_int(state.key_root + "0")
        scale = md.get_key_ints(state.key_root, state.scale_name)
        self.scale_notes = sorted([tonic + d + 12*k for k in range(-2,8) for d in scale])
        # Instruments
        self.drums = DrumKit(self.osc, state)
        self.voices = {
            "zone1": MelodicVoice("zone1", self.osc, state),
            "zone2": MelodicVoice("zone2", self.osc, state),
            "zone3": MelodicVoice("zone3", self.osc, state),
        }
        self.current_chord_notes: List[int] = []
        self.current_chord_symbol: str = "C_M"

    def choose_and_voice_chord(self):
        max_t = max(self.state.zone_tension.values()) if self.state.zone_tension else 0.0
        symbol = self.harmony.next_chord(max_t)
        base_notes = HarmonyEngine.chord_to_midi(symbol)  # e.g., [C0, E0, G0] or [C0,E0,G0,Bb0]
        base_notes = sorted(base_notes)

        # Pick distinct chord degrees for the three zones:
        # - for triads: root, third, fifth
        # - for 7ths+: root, third, seventh (keeps color on top)
        if len(base_notes) >= 4:
            degree_set = [base_notes[0], base_notes[1], base_notes[3]]  # R,3,7
        elif len(base_notes) == 3:
            degree_set = [base_notes[0], base_notes[1], base_notes[2]]  # R,3,5
        elif len(base_notes) == 2:
            degree_set = [base_notes[0], base_notes[1], base_notes[0]]  # duplicate if needed
        else:
            degree_set = [base_notes[0], base_notes[0], base_notes[0]]  # single-note chord

        def transpose_near(note0: int, center: int) -> int:
            # choose note0 + 12k nearest to center
            candidates = [note0 + 12 * k for k in range(-6, 7)]
            return min(candidates, key=lambda n: abs(n - center))

        # Map degrees to zones 1..3 uniquely
        z1_target = transpose_near(degree_set[0], self.state.zone_registers["zone1"])  # root -> low
        z2_target = transpose_near(degree_set[1], self.state.zone_registers["zone2"])  # 3rd -> mid
        z3_target = transpose_near(degree_set[2], self.state.zone_registers["zone3"])  # 5th/7th -> high

        self.current_chord_notes = [z1_target, z2_target, z3_target]
        self.current_chord_symbol = symbol

        # sustain for one chord window (full bar or chord_span)
        beats = max(1, self.state.chord_span_beats)  # or self.state.beats_per_bar if you want exactly a bar
        spb = 60.0 / max(20.0, min(240.0, self.state.bpm))
        dur_sec = float(spb * beats) - 0.025

        self.osc(self.state.ports["zone1"], "/note", [z1_target, 0.9, dur_sec, self.state.channels["zone1"]])
        self.osc(self.state.ports["zone2"], "/note", [z2_target, 0.7, dur_sec, self.state.channels["zone2"]])
        self.osc(self.state.ports["zone3"], "/note", [z3_target, 0.7, dur_sec, self.state.channels["zone3"]])

    async def run(self):
        step16 = 0
        chord_span = max(1, self.state.chord_span_beats)
        # seed chord
        self.choose_and_voice_chord()

        async for beat, sub in self.clock.ticker(sub_per_beat=4):
            if not self.state.running: break
            # drums at 4th
            if step16 % 4 == 0:
                self.drums.step(step16 // 4)
            step16 = (step16 + 1) % 16
            # chord change
            if (beat % chord_span == 0) and (sub == 0):
                self.choose_and_voice_chord()
            # melodic zones on 8ths
            for v in self.voices.values():
                v.melodic_step(sub, self.current_chord_notes, self.scale_notes)

# ---- CLI demo / live controls ----

async def main():
    st = State()
    cond = Conductor(st)

    async def stdin_commands():
        loop = asyncio.get_running_loop()
        while st.running:
            try:
                cmd = await loop.run_in_executor(None, input, "")
            except (EOFError, KeyboardInterrupt):
                st.running = False
                break
            parts = cmd.strip().split()
            if not parts: continue
            if parts[0] == "fill":
                cond.drums.queue_fill()
                print("[drums] queued fill")
            elif parts[0] == "bpm" and len(parts) > 1:
                st.bpm = float(parts[1]); print(f"[bpm] -> {st.bpm}")
            elif parts[0] == "t" and len(parts) == 3:
                zone, val = parts[1], float(parts[2])
                if zone in st.zone_tension:
                    st.zone_tension[zone] = max(0.0, min(1.0, val))
                    print(f"[tension] {zone} -> {st.zone_tension[zone]:.2f}")
            elif parts[0] in ("quit","exit"):
                st.running = False
            else:
                print("commands: fill | bpm 128 | t zone1 0.7 | t zone2 0.3 | t zone3 0.9 | exit")

    print(f"Connecting to MOSC on ports: {st.ports}")
    await asyncio.gather(cond.run(), stdin_commands())

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass