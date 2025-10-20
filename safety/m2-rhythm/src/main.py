#!/usr/bin/env python3
"""
OSC Rhythm Pulse Sender
-----------------------
Sends MIDI note impulses over OSC to your JUCE plugin's OSC→MIDI bridge.
You can switch rhythms from the terminal by pressing 1–4. The change takes effect only
after the current rhythm finishes its loop. Press 'q' to quit.

Usage:
  python rhythm_osc.py --ip 127.0.0.1 --port 9000 --bpm 120 --pulse-width 0.02
  python rhythm_osc.py --ip 127.0.0.1 --port 9000 --bpm 120 --pulse-width 0.02 --midi-note 60 --velocity 0.9 --channel 1

Requires:
  pip install python-osc

Notes:
- MIDI impulses are sent as `/note <note> <velocity01> <durationSec> [channel]` (default address `/note`).
- Rhythms are defined as sequences of IOIs (inter-onset intervals) in *beats*;
  the sum of each pattern equals one bar. BPM maps beats -> seconds.
"""

import argparse
import sys
import threading
import time
import signal
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pythonosc.udp_client import SimpleUDPClient


# ---------------------- Rhythm Definitions ----------------------
PATTERNS_BEATS: Dict[str, List[float]] = {
    "1": [1.0] * 4,                         # Straight 8ths (8 pulses per bar)
    "2": [0.75, 0.75, 0.5],  # [1.5, 1.5, 1.0],                   # Tresillo (3-3-2) in beats
    # A 3-2 son clave mapped on 16th grid over 4 beats (hits at: 1, 1e&, 3, 4&, 4):
    # IOIs in beats between hits: [1.0, 0.5, 1.5, 0.5, 0.5] -> sums to 4.0
    "3": [1.0, 0.5, 1.5, 0.5, 0.5],
    "4": [1.0, 0.5, 1.5, 1.0, 1.0, 2.5, 0.5], # 2 5 1 2 1 3
}

DEFAULT_PATTERN_KEY = "1"

# ---------------------- Engine ----------------------

@dataclass
class RhythmEngine:
    ip: str
    port: int
    bpm: float
    pulse_width: float
    midi_note: int
    velocity01: float
    channel: int
    address: str = "/note"
    client: SimpleUDPClient = field(init=False)
    stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    pattern_key: str = field(default=DEFAULT_PATTERN_KEY, init=False)
    pending_pattern_key: Optional[str] = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self):
        self.client = SimpleUDPClient(self.ip, self.port)

    # Convert beats to seconds at current bpm
    def beats_to_seconds(self, beats: float) -> float:
        return (60.0 / self.bpm) * beats

    def set_pending_pattern(self, key: str):
        if key not in PATTERNS_BEATS:
            print(f"Unknown pattern '{key}'. Valid options: {', '.join(PATTERNS_BEATS.keys())}")
            return
        with self._lock:
            self.pending_pattern_key = key
        print(f"[queued] Will switch to pattern {key} after current loop finishes.")

    def apply_pending_if_any(self):
        with self._lock:
            if self.pending_pattern_key is not None:
                self.pattern_key = self.pending_pattern_key
                self.pending_pattern_key = None
                print(f"[switched] Now playing pattern {self.pattern_key}.")

    def pulse(self):
        # Send a MIDI impulse via the JUCE bridge: /note <int note> <float vel01> <float duration> [int channel]
        payload = [int(self.midi_note), float(self.velocity01), float(self.pulse_width), int(self.channel)]
        print(f"Sending payload @ {self.port}:", payload)
        self.client.send_message(self.address, payload)

    def play_loop(self):
        print(f"Sending OSC to {self.ip}:{self.port} at {self.address}")
        print(f"Starting with pattern {self.pattern_key} @ {self.bpm} BPM → MIDI note {self.midi_note} (vel {self.velocity01:.2f}, ch {self.channel}), duration {self.pulse_width:.3f}s")
        print("Controls: [1]-[4] patterns | bpm <num> | pw <sec> | note <0-127> | vel <0..1> | chan <1-16> | q to quit\n")

        while not self.stop_event.is_set():
            pattern = PATTERNS_BEATS[self.pattern_key]

            # One full bar according to the current pattern
            for ioi_beats in pattern:
                if self.stop_event.is_set():
                    break
                # Pulse:
                start_time = time.time()
                self.pulse()
                # Sleep the remainder of the IOI minus pulse width (clipped)
                ioi_seconds = self.beats_to_seconds(ioi_beats)
                remainder = max(0.0, ioi_seconds - self.pulse_width)
                # Busy-wait less; use sleep:
                time.sleep(remainder)

            # Only after finishing the pattern do we apply the queued change
            self.apply_pending_if_any()

        print("Stopping…")

    def request_stop(self):
        self.stop_event.set()

# ---------------------- Interaction ----------------------

def input_thread(engine: RhythmEngine):
    # Separate thread: read single-line commands
    while not engine.stop_event.is_set():
        try:
            cmd = input().strip().lower()
        except EOFError:
            engine.request_stop()
            break
        except KeyboardInterrupt:
            engine.request_stop()
            break

        if cmd in ("1", "2", "3", "4", "5"):
            engine.set_pending_pattern(cmd)
        elif cmd == "q":
            engine.request_stop()
        elif cmd.startswith("bpm"):
            parts = cmd.split()
            if len(parts) == 2:
                try:
                    new_bpm = float(parts[1])
                    engine.bpm = max(1.0, new_bpm)
                    print(f"[tempo] BPM set to {engine.bpm}")
                except ValueError:
                    print("Usage: bpm <number>")
            else:
                print("Usage: bpm <number>")
        elif cmd.startswith("pw"):
            parts = cmd.split()
            if len(parts) == 2:
                try:
                    new_pw = float(parts[1])
                    engine.pulse_width = max(0.001, new_pw)
                    print(f"[pulse] Width set to {engine.pulse_width:.3f}s")
                except ValueError:
                    print("Usage: pw <seconds>")
            else:
                print("Usage: pw <seconds>")
        elif cmd.startswith("note"):
            parts = cmd.split()
            if len(parts) == 2:
                try:
                    n = int(parts[1])
                    if 0 <= n <= 127:
                        engine.midi_note = n
                        print(f"[midi] Note set to {engine.midi_note}")
                    else:
                        print("Note must be 0..127")
                except ValueError:
                    print("Usage: note <0-127>")
            else:
                print("Usage: note <0-127>")
        elif cmd.startswith("vel"):
            parts = cmd.split()
            if len(parts) == 2:
                try:
                    v = float(parts[1])
                    if 0.0 <= v <= 1.0:
                        engine.velocity01 = v
                        print(f"[midi] Velocity set to {engine.velocity01:.2f}")
                    else:
                        print("Velocity must be between 0.0 and 1.0")
                except ValueError:
                    print("Usage: vel <0..1>")
            else:
                print("Usage: vel <0..1>")
        elif cmd.startswith("chan"):
            parts = cmd.split()
            if len(parts) == 2:
                try:
                    c = int(parts[1])
                    if 1 <= c <= 16:
                        engine.channel = c
                        print(f"[midi] Channel set to {engine.channel}")
                    else:
                        print("Channel must be 1..16")
                except ValueError:
                    print("Usage: chan <1-16>")
            else:
                print("Usage: chan <1-16>")
        else:
            print("Commands: 1-5 | bpm <num> | pw <sec> | note <0-127> | vel <0..1> | chan <1-16> | q")

def main():
    parser = argparse.ArgumentParser(description="OSC rhythm pulse sender")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="OSC destination IP")
    parser.add_argument("--port", type=int, default=20000, help="OSC destination port")
    parser.add_argument("--bpm", type=float, default=120.0, help="Tempo in BPM (quarter-note = 1 beat)")
    parser.add_argument("--pulse-width", type=float, default=0.1, help="Seconds to hold '1' before returning to '0'")
    parser.add_argument("--address", type=str, default="/note", help="OSC address for pulses")
    parser.add_argument("--midi-note", type=int, default=51, help="MIDI note number (0-127)")
    parser.add_argument("--velocity", type=float, default=1.0, help="MIDI velocity in 0..1")
    parser.add_argument("--channel", type=int, default=1, help="MIDI channel (1-16)")
    args = parser.parse_args()

    engine = RhythmEngine(
        ip=args.ip, port=args.port, bpm=args.bpm, pulse_width=args.pulse_width, address=args.address,
        midi_note=args.midi_note, velocity01=args.velocity, channel=args.channel
    )

    # Handle force quit
    def _sigint(_sig, _frame):
        engine.request_stop()
    signal.signal(signal.SIGINT, _sigint)

    t_in = threading.Thread(target=input_thread, args=(engine,), daemon=True)
    t_in.start()

    try:
        engine.play_loop()
    finally:
        engine.request_stop()
        t_in.join(timeout=0.5)

if __name__ == "__main__":
    main()
