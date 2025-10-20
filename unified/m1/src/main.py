# main.py
import asyncio
from orchestrator import GlobalConfig, PortRegistry, Orchestrator, md

cfg = GlobalConfig(
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
    }
)

ports = PortRegistry({
    "drums": 9000,
    "harm.voice1": 9001,
    "harm.voice2": 9002,
    "harm.voice3": 9003,
    "lead": 9004
})

orch = Orchestrator(cfg, ports)

# Optional: realtime OSC middleware (e.g., global velocity trim, metering, logging)
orch.add_middleware(lambda msg: msg)  # no-op

# Instruments
drums = orch.add_percussion("drums")
harm  = orch.add_harmonic_group("harm", ["harm.voice1","harm.voice2","harm.voice3"])
lead  = orch.add_melodic("lead")

# Live controls (simulate your CLI)
async def live_control():
    # Set initial zone tensions (these drive chord selection by default)
    orch.set_tension("zone1", 0.2)
    orch.set_tension("zone2", 0.4)
    orch.set_tension("zone3", 0.7)

    # After 4 seconds, queue a 3-beat fill that starts exactly 3 beats before the next bar
    await asyncio.sleep(4.0)
    drums.queue_fill_beats(fill_len_beats=3, preset="toms")

    # Force next bar chord override:
    await asyncio.sleep(2.0)
    orch.queue_next_chord("harm", "F_M7")

    # Fire a few melodic notes manually:
    await asyncio.sleep(1.0)
    lead.play(md.note_to_int("A4"), 0.8, 0.12)
    await asyncio.sleep(0.15)
    lead.play(md.note_to_int("B4"), 0.85, 0.1)

async def main():
    await asyncio.gather(orch.run(), live_control())

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        orch.stop()
