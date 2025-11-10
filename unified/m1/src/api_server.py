# api_server.py
import asyncio
import logging
from typing import Dict, List, Set
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

from orchestrator import GlobalConfig, PortRegistry, Orchestrator, md

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- Request/Response Models ----

class TensionUpdate(BaseModel):
    zone1: float = Field(ge=0.0, le=1.0)
    zone2: float = Field(ge=0.0, le=1.0)
    zone3: float = Field(ge=0.0, le=1.0)

class BPMUpdate(BaseModel):
    bpm: float = Field(ge=40.0, le=240.0)

class KeyUpdate(BaseModel):
    key: str
    
    @validator('key')
    def validate_key(cls, v):
        valid_keys = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
        if v not in valid_keys:
            raise ValueError(f"Key must be one of {valid_keys}")
        return v

class BeatsPerBarUpdate(BaseModel):
    beats_per_bar: int = Field(ge=1, le=12)

class ChordOverride(BaseModel):
    group_id: str = "harm"
    chord_symbol: str
    
    @validator('chord_symbol')
    def validate_chord(cls, v):
        if "_" not in v:
            raise ValueError("Chord symbol must contain '_' (e.g., 'C_M', 'F_M7')")
        return v

class DrumFill(BaseModel):
    preset: str = Field(default="snare")
    beats: int = Field(ge=1, le=32, default=3)
    
    @validator('preset')
    def validate_preset(cls, v):
        if v not in ["snare", "toms", "hats"]:
            raise ValueError("Preset must be one of: snare, toms, hats")
        return v

class OrchestratorState(BaseModel):
    bpm: float
    beats_per_bar: int
    key_root: str
    tensions: Dict[str, float]
    current_bar: int = 0

# ---- WebSocket Manager ----

class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to websocket: {e}")
                disconnected.add(connection)
        
        for conn in disconnected:
            self.disconnect(conn)

# ---- Orchestrator Wrapper ----

class OrchestratorService:
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
            "harm.voice1": 9002,
            "harm.voice2": 9004,
            "harm.voice3": 9003,
            "lead": 9004,
            "drone_a": 8000,
            "drone_b": 8001,
        })
        
        self.orch: Orchestrator = None
        self.manager = ConnectionManager()
        self.current_bar = 0
        self._lock = asyncio.Lock()
        self._running = False
    
    async def start(self):
        if self._running:
            return
        
        self._running = True
        self.orch = Orchestrator(self.cfg, self.ports)
        self.orch.add_middleware(lambda msg: msg)
        
        # Create instruments
        self.orch.add_percussion("drums")
        self.orch.add_harmonic_group("harm", ["harm.voice1", "harm.voice2", "harm.voice3"])
        self.orch.add_melodic("lead")
        
        # Seed initial tensions
        self.orch.set_tension("zone1", 0.2)
        self.orch.set_tension("zone2", 0.4)
        self.orch.set_tension("zone3", 0.7)
        
        # Hook into scheduler for bar updates
        self.orch._sched.on_bar(self._on_bar)
        
        # Start orchestrator in background
        asyncio.create_task(self.orch.run())
        logger.info("Orchestrator service started")
    
    async def stop(self):
        if not self._running:
            return
        
        self._running = False
        if self.orch:
            self.orch.stop()
        logger.info("Orchestrator service stopped")
    
    def _on_bar(self, bar_idx: int):
        self.current_bar = bar_idx
        asyncio.create_task(self.manager.broadcast({
            "type": "bar_update",
            "bar": bar_idx
        }))
    
    async def set_tensions(self, tensions: TensionUpdate):
        async with self._lock:
            self.orch.set_tension("zone1", tensions.zone1)
            self.orch.set_tension("zone2", tensions.zone2)
            self.orch.set_tension("zone3", tensions.zone3)
        
        await self.manager.broadcast({
            "type": "tension_update",
            "tensions": {
                "zone1": tensions.zone1,
                "zone2": tensions.zone2,
                "zone3": tensions.zone3
            }
        })
    
    async def set_bpm(self, bpm: float):
        async with self._lock:
            self.orch.set_bpm(bpm)
            self.cfg.bpm = bpm
        
        await self.manager.broadcast({
            "type": "bpm_update",
            "bpm": bpm
        })
    
    async def set_key(self, key: str):
        async with self._lock:
            self.cfg.key_root = key
            for inst in self.orch._harm_groups.values():
                from melody import HarmonyGenerator
                inst.hg = HarmonyGenerator(key=key)
        
        await self.manager.broadcast({
            "type": "key_update",
            "key": key
        })
    
    async def set_beats_per_bar(self, bpb: int):
        async with self._lock:
            self.cfg.beats_per_bar = bpb
        
        await self.manager.broadcast({
            "type": "beats_per_bar_update",
            "beats_per_bar": bpb
        })
    
    async def queue_chord(self, group_id: str, symbol: str):
        async with self._lock:
            self.orch.queue_next_chord(group_id, symbol)
        
        await self.manager.broadcast({
            "type": "chord_queued",
            "chord": symbol
        })
    
    async def trigger_fill(self, preset: str, beats: int):
        async with self._lock:
            drums = self.orch._percussion.get("drums")
            if drums:
                drums.queue_fill_beats(beats, preset=preset)
        
        await self.manager.broadcast({
            "type": "fill_triggered",
            "preset": preset,
            "beats": beats
        })
    
    def get_state(self) -> OrchestratorState:
        return OrchestratorState(
            bpm=self.cfg.bpm,
            beats_per_bar=self.cfg.beats_per_bar,
            key_root=self.cfg.key_root,
            tensions={
                "zone1": self.orch._tension.get("zone1", 0.0),
                "zone2": self.orch._tension.get("zone2", 0.0),
                "zone3": self.orch._tension.get("zone3", 0.0),
            },
            current_bar=self.current_bar
        )

# ---- FastAPI App ----

service = OrchestratorService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await service.start()
    yield
    # Shutdown
    await service.stop()

app = FastAPI(title="Orchestrator API", version="1.0.0", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- HTTP Endpoints ----

@app.get("/")
async def root():
    return {"message": "Orchestrator API", "status": "running"}

@app.get("/state", response_model=OrchestratorState)
async def get_state():
    return service.get_state()

@app.post("/tension")
async def update_tension(tension: TensionUpdate):
    try:
        await service.set_tensions(tension)
        return {"status": "ok", "tensions": tension.dict()}
    except Exception as e:
        logger.error(f"Error updating tension: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/bpm")
async def update_bpm(bpm_update: BPMUpdate):
    try:
        await service.set_bpm(bpm_update.bpm)
        return {"status": "ok", "bpm": bpm_update.bpm}
    except Exception as e:
        logger.error(f"Error updating BPM: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/key")
async def update_key(key_update: KeyUpdate):
    try:
        await service.set_key(key_update.key)
        return {"status": "ok", "key": key_update.key}
    except Exception as e:
        logger.error(f"Error updating key: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/beats_per_bar")
async def update_beats_per_bar(bpb_update: BeatsPerBarUpdate):
    try:
        await service.set_beats_per_bar(bpb_update.beats_per_bar)
        return {"status": "ok", "beats_per_bar": bpb_update.beats_per_bar}
    except Exception as e:
        logger.error(f"Error updating beats per bar: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chord")
async def queue_chord(chord: ChordOverride):
    try:
        await service.queue_chord(chord.group_id, chord.chord_symbol)
        return {"status": "ok", "chord": chord.chord_symbol}
    except Exception as e:
        logger.error(f"Error queuing chord: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fill")
async def trigger_fill(fill: DrumFill):
    try:
        await service.trigger_fill(fill.preset, fill.beats)
        return {"status": "ok", "preset": fill.preset, "beats": fill.beats}
    except Exception as e:
        logger.error(f"Error triggering fill: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ---- WebSocket Endpoint ----

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await service.manager.connect(websocket)
    
    try:
        # Send initial state
        state = service.get_state()
        await websocket.send_json({
            "type": "initial_state",
            "state": state.dict()
        })
        
        # Keep connection alive
        while True:
            data = await websocket.receive_text()
            # Echo back for heartbeat
            await websocket.send_json({"type": "pong"})
    
    except WebSocketDisconnect:
        service.manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        service.manager.disconnect(websocket)

# ---- Main ----

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
