import asyncio, time, random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterable, Any, Union
import common.midi as md
from melody import HarmonyGenerator
from common.osc import OSCManager

Scalar = Union[int, float, str, bytes, bool]
Arg = Union[Scalar, Iterable[Scalar]]

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

        # Base hit strengths; scaled by global tension below
        self.base_vel = {"kick":0.85, "snare":0.8, "chh":0.6, "ohh":0.7, "clap":0.8, "ride":0.9, "tom":0.8}

    def start_fill(self, measures: int = 1, preset: str = "snare_roll"):
        beats = max(1, measures) * self.state.beats_per_bar
        steps: List[Dict[str, int]] = []
        if preset == "snare_roll":
            for _ in range(beats - 1):
                steps.append({"snare": 1})
            steps.append({"ride": 1, "snare": 1})
        elif preset == "toms":
            for _ in range(beats - 1):
                steps.append({"tom": 1})
            steps.append({"ride": 1, "snare": 1})
        elif preset == "hats":
            for i in range(beats - 1):
                steps.append({"chh": 1})
            steps.append({"ride": 1})
        else:
            for _ in range(beats - 1):
                steps.append({"snare": 1})
            steps.append({"ride": 1, "snare": 1})
        self.fill_queue = steps

    def queue_fill(self, steps: Optional[List[Dict[str,int]]] = None):
        # default 1 bar snare fill if not provided
        if steps is None:
            self.start_fill(measures=1, preset="snare_roll")
        else:
            self.fill_queue = steps

    def step(self, step_beats: int):
        port = self.state.ports["drums"]
        ch   = self.state.channels["drums"]

        pat = self.fill_queue.pop(0) if self.fill_queue else self.pattern[step_beats % len(self.pattern)]
        if self.fill_queue == []:
            self.fill_queue = None

        # Global (max) tension to scale velocities
        t_global = max(self.state.zone_tension.values()) if self.state.zone_tension else 0.0
        vel_scale = 0.5 + 0.5 * t_global  # 0.5..1.0

        for name, on in pat.items():
            if on:
                nn = self.state.drum_map.get(name, 36)
                v0 = self.base_vel.get(name, 0.8)
                vel = min(1.0, v0 * vel_scale)
                self.osc(port, "/note", [nn, vel, 0.05, ch])

class MelodicVoice:
    def __init__(self, zone_key: str, osc: OSCManager, state: State):
        self.zone, self.osc, self.state = zone_key, osc, state
        self.last_note: Optional[int] = None
        self.hold_until: float = 0.0
        self.anchor_note: Optional[int] = None  # set on chord changes

    def set_anchor(self, note: int):
        self.anchor_note = note
        # pull toward anchor at chord change to keep identity
        self.last_note = note

    def melodic_step(self, sub: int, chord_notes: List[int], scale_notes: List[int]):
        # fire only when tensioned
        t = float(self.state.zone_tension.get(self.zone, 0.0))
        if t < 1.0:
            return
        # fire on 8ths
        if sub % 2:
            return

        # density grows with tension
        density = 0.3 + 0.7 * t
        if random.random() > density:
            return

        now = time.perf_counter()
        if now < self.hold_until:
            return

        center = self.anchor_note if self.anchor_note is not None else self.state.zone_registers[self.zone]

        # tension-dependent hover radius (small at low t, wider at high t)
        r = 2 + int(6 * t)  # ~2..8 semitones
        step_choices = list(range(-r, r + 1))
        base = self.last_note if self.last_note is not None else center
        candidate = base + random.choice(step_choices)

        # mean-reversion: pull toward center
        if random.random() < 0.65:
            candidate = int(round(candidate + 0.33 * (center - candidate)))

        # quantize to scale
        target = min(scale_notes, key=lambda n: abs(n - candidate))

        # optionally bias toward chord field
        if random.random() < (0.6 + 0.3 * t):
            chord_field = [p + 12 * k for p in chord_notes for k in range(-2, 3)]
            target = min(chord_field, key=lambda n: abs(n - target))

        port = self.state.ports[self.zone]
        ch   = self.state.channels[self.zone]

        # velocity proportional to this zone’s tension
        vel = 0.4 + 0.6 * t   # 0.4..1.0
        dur = 0.10 if t < 0.5 else 0.08

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
        # - for 7ths+: root, third, seventh
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

        # set melodic anchors so each voice “hovers” around its chord degree
        self.voices["zone1"].set_anchor(z1_target)
        self.voices["zone2"].set_anchor(z2_target)
        self.voices["zone3"].set_anchor(z3_target)

        # sustain for one chord window (full bar or chord_span)
        beats = max(1, self.state.chord_span_beats)
        spb = 60.0 / max(20.0, min(240.0, self.state.bpm))
        dur_sec = float(spb * beats) - 0.025

        # chord sustain velocities scale with global tension
        t_global = max(self.state.zone_tension.values()) if self.state.zone_tension else 0.0
        v_low  = 0.6 + 0.4 * t_global
        v_mid  = 0.5 + 0.4 * t_global
        v_high = 0.5 + 0.4 * t_global

        self.osc(self.state.ports["zone1"], "/note", [z1_target, v_low,  dur_sec, self.state.channels["zone1"]])
        self.osc(self.state.ports["zone2"], "/note", [z2_target, v_mid,  dur_sec, self.state.channels["zone2"]])
        self.osc(self.state.ports["zone3"], "/note", [z3_target, v_high, dur_sec, self.state.channels["zone3"]])

    async def run(self):
        step16 = 0
        chord_span = max(1, self.state.chord_span_beats)
        # seed chord
        self.choose_and_voice_chord()

        async for beat, sub in self.clock.ticker(sub_per_beat=4):
            if not self.state.running:
                break
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
            if not parts:
                continue

            # --- DRUM FILLS ---
            if parts[0] == "fill":
                # fill            -> 1-bar snare
                # fill 2          -> 2-bar snare
                # fill toms       -> 1-bar toms
                # fill hats       -> 1-bar hats
                # fill snare 3    -> 3-bar snare
                preset = "snare_roll"
                measures = 1
                if len(parts) >= 2:
                    if parts[1].isdigit():
                        measures = int(parts[1])
                    elif parts[1] in ("toms", "hats", "snare"):
                        preset = "toms" if parts[1] == "toms" else ("hats" if parts[1] == "hats" else "snare_roll")
                        if len(parts) >= 3 and parts[2].isdigit():
                            measures = int(parts[2])
                cond.drums.start_fill(measures=measures, preset=preset)
                print(f"[drums] {measures}-bar fill preset={preset}")
                continue

            if parts[0] == "fill1":
                cond.drums.start_fill(measures=1, preset="snare_roll")
                print("[drums] 1-bar fill (snare)")
                continue

            if parts[0] == "fill2":
                cond.drums.start_fill(measures=2, preset="snare_roll")
                print("[drums] 2-bar fill (snare)")
                continue

            # --- TEMPO ---
            if parts[0] == "bpm" and len(parts) > 1:
                try:
                    st.bpm = float(parts[1])
                    print(f"[bpm] -> {st.bpm}")
                except ValueError:
                    print("[bpm] expected a number, e.g., bpm 128")
                continue

            # --- TENSION PER ZONE ---
            if parts[0] == "t" and len(parts) == 3:
                zone, val_str = parts[1], parts[2]
                try:
                    val = float(val_str)
                except ValueError:
                    print("[tension] value must be a float 0..1")
                    continue
                if zone in st.zone_tension:
                    st.zone_tension[zone] = max(0.0, min(1.0, val))
                    print(f"[tension] {zone} -> {st.zone_tension[zone]:.2f}")
                else:
                    print(f"[tension] unknown zone '{zone}', expected one of: {list(st.zone_tension.keys())}")
                continue

            # --- EXIT ---
            if parts[0] in ("quit", "exit"):
                st.running = False
                continue

            print("commands:\n"
                  "  fill | fill N | fill snare|toms|hats [N]\n"
                  "  fill1 | fill2\n"
                  "  bpm 128\n"
                  "  t zone1 0.7 | t zone2 0.3 | t zone3 0.9\n"
                  "  exit")

    print(f"Connecting to MOSC on ports: {st.ports}")
    await asyncio.gather(cond.run(), stdin_commands())

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
