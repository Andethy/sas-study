#!/usr/bin/env python3
"""
OSC Rhythm Pulse Sender
-----------------------
Sends brief pulses (1 then 0) to an OSC address (/rhythm) following one of three preset rhythms.
You can switch rhythms from the terminal by pressing 1, 2, or 3. The change takes effect only
after the current rhythm finishes its loop. Press 'q' to quit.

Usage:
  python rhythm_osc.py --ip 127.0.0.1 --port 9000 --bpm 120 --pulse-width 0.02

Requires:
  pip install python-osc

Notes:
- Pulses are sent as 1 followed by 0 at /rhythm. Pulse width is in seconds.
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
    "2": [1.5, 1.5, 1.0],                   # Tresillo (3-3-2) in beats
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
    address: str = "/rhythm"
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
        # Send 1 then 0 separated by pulse_width
        self.client.send_message(self.address, 1.0)
        time.sleep(self.pulse_width)
        self.client.send_message(self.address, 0.0)

    def play_loop(self):
        print(f"Sending OSC to {self.ip}:{self.port} at {self.address}")
        print(f"Starting with pattern {self.pattern_key} @ {self.bpm} BPM. Pulse width: {self.pulse_width:.3f}s")
        print("Controls: [1]-[4] to queue pattern change (applies after bar), 'q' to quit.\n")

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
        else:
            print("Commands: 1-5 | bpm <num> | pw <sec> | q")

def main():
    parser = argparse.ArgumentParser(description="OSC rhythm pulse sender")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="OSC destination IP")
    parser.add_argument("--port", type=int, default=9000, help="OSC destination port")
    parser.add_argument("--bpm", type=float, default=120.0, help="Tempo in BPM (quarter-note = 1 beat)")
    parser.add_argument("--pulse-width", type=float, default=0.1, help="Seconds to hold '1' before returning to '0'")
    parser.add_argument("--address", type=str, default="/rhythm", help="OSC address for pulses")
    args = parser.parse_args()

    engine = RhythmEngine(
        ip=args.ip, port=args.port, bpm=args.bpm, pulse_width=args.pulse_width, address=args.address
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
