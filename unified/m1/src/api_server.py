# api_server.py
import asyncio
import logging
from typing import Dict, List, Set
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, validator
import uvicorn
import os
import shutil
import subprocess
import tempfile

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

class TimbreMixRequest(BaseModel):
    mix_value: float = Field(ge=0.0, le=1.0)

class TimbreUploadResponse(BaseModel):
    filename: str
    file_id: str
    status: str

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
        
        # Timbre interpolation state
        self.timbre_files = {"sample_a": None, "sample_b": None}
        self.current_mix = 0.5
        self.upload_dir = "uploads"
        os.makedirs(self.upload_dir, exist_ok=True)
    
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
    
    async def upload_timbre_file(self, file: UploadFile, sample_type: str) -> TimbreUploadResponse:
        """Upload and store a timbre sample file"""
        if sample_type not in ["sample_a", "sample_b"]:
            raise ValueError("sample_type must be 'sample_a' or 'sample_b'")
        
        # Generate unique filename
        file_extension = os.path.splitext(file.filename)[1] if file.filename else ".wav"
        file_id = f"{sample_type}_{int(asyncio.get_event_loop().time())}{file_extension}"
        file_path = os.path.join(self.upload_dir, file_id)
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Store file info
        self.timbre_files[sample_type] = {
            "filename": file.filename,
            "file_id": file_id,
            "file_path": file_path
        }
        
        logger.info(f"Uploaded {sample_type}: {file.filename} -> {file_path}")
        
        return TimbreUploadResponse(
            filename=file.filename or "unknown",
            file_id=file_id,
            status="uploaded"
        )
    
    async def set_timbre_mix(self, mix_value: float):
        """Set the timbre interpolation mix value and call m1-timbre"""
        self.current_mix = max(0.0, min(1.0, mix_value))
        
        # Check if both files are uploaded
        if not (self.timbre_files["sample_a"] and self.timbre_files["sample_b"]):
            raise ValueError("Both sample A and sample B must be uploaded before mixing")
        
        # Call m1-timbre app to perform interpolation
        try:
            sample_a_path = self.timbre_files["sample_a"]["file_path"]
            sample_b_path = self.timbre_files["sample_b"]["file_path"]
            
            # Create output path
            output_path = os.path.join(self.upload_dir, f"mixed_{int(asyncio.get_event_loop().time())}.wav")
            
            # Call m1-timbre subprocess from the correct working directory
            # Calculate path to m1-timbre directory
            # Get the current file's directory, go up to repo root, then to safety/m1-timbre/src
            current_dir = os.path.dirname(os.path.abspath(__file__))  # unified/m1/src
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))  # go up 3 levels to repo root
            m1_timbre_dir = os.path.join(repo_root, "safety", "m1-timbre", "src")
            
            # Alternative path calculation if the first one doesn't work
            if not os.path.exists(m1_timbre_dir):
                # Try relative to current working directory
                alt_m1_timbre_dir = os.path.join(os.getcwd(), "safety", "m1-timbre", "src")
                if os.path.exists(alt_m1_timbre_dir):
                    m1_timbre_dir = alt_m1_timbre_dir
                    logger.info(f"Using alternative path: {m1_timbre_dir}")
                else:
                    # Try going up from current working directory
                    cwd_parent = os.path.dirname(os.getcwd())
                    alt_m1_timbre_dir2 = os.path.join(cwd_parent, "safety", "m1-timbre", "src")
                    if os.path.exists(alt_m1_timbre_dir2):
                        m1_timbre_dir = alt_m1_timbre_dir2
                        logger.info(f"Using alternative path 2: {m1_timbre_dir}")
            
            # Debug: log the path calculation
            logger.info(f"Current working directory: {os.getcwd()}")
            logger.info(f"Current file dir: {current_dir}")
            logger.info(f"Repo root: {repo_root}")
            logger.info(f"Looking for m1-timbre at: {m1_timbre_dir}")
            logger.info(f"m1-timbre directory exists: {os.path.exists(m1_timbre_dir)}")
            
            # Also check if main.py exists in the directory
            main_py_path = os.path.join(m1_timbre_dir, "main.py")
            logger.info(f"main.py exists at {main_py_path}: {os.path.exists(main_py_path)}")
            
            cmd = [
                "python", "-m", "main",
                "--sample-a", os.path.abspath(sample_a_path),
                "--sample-b", os.path.abspath(sample_b_path),
                "--mix", str(self.current_mix),
                "--output", os.path.abspath(output_path),
                "--headless"
            ]
            
            # Verify the directory exists
            if not os.path.exists(m1_timbre_dir):
                raise FileNotFoundError(f"m1-timbre directory not found: {m1_timbre_dir}")
            
            logger.info(f"Calling m1-timbre from {m1_timbre_dir}: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=m1_timbre_dir)
            
            if result.returncode == 0:
                logger.info(f"Timbre interpolation successful: {output_path}")
                
                # Broadcast update to connected clients
                await self.manager.broadcast({
                    "type": "timbre_mix_update",
                    "mix_value": self.current_mix,
                    "output_file": output_path,
                    "status": "success"
                })
                
                # Create audio URL for frontend
                filename = os.path.basename(output_path)
                audio_url = f"/audio/{filename}"
                
                return {
                    "status": "success", 
                    "mix_value": self.current_mix, 
                    "output_file": output_path,
                    "audio_url": audio_url
                }
            else:
                error_msg = f"m1-timbre failed: {result.stderr}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
                
        except subprocess.TimeoutExpired:
            raise RuntimeError("Timbre interpolation timed out")
        except Exception as e:
            logger.error(f"Timbre interpolation error: {e}")
            raise RuntimeError(f"Timbre interpolation failed: {str(e)}")
    
    def get_timbre_status(self):
        """Get current timbre interpolation status"""
        return {
            "sample_a": self.timbre_files["sample_a"]["filename"] if self.timbre_files["sample_a"] else None,
            "sample_b": self.timbre_files["sample_b"]["filename"] if self.timbre_files["sample_b"] else None,
            "current_mix": self.current_mix,
            "ready": bool(self.timbre_files["sample_a"] and self.timbre_files["sample_b"])
        }

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

# ---- Timbre Interpolation Endpoints ----

@app.post("/timbre/upload/{sample_type}")
async def upload_timbre_sample(sample_type: str, file: UploadFile = File(...)):
    """Upload a timbre sample file (sample_a or sample_b)"""
    try:
        if sample_type not in ["sample_a", "sample_b"]:
            raise HTTPException(status_code=400, detail="sample_type must be 'sample_a' or 'sample_b'")
        
        # Validate file type
        if not file.filename or not any(file.filename.lower().endswith(ext) for ext in ['.wav', '.mp3', '.flac', '.aiff']):
            raise HTTPException(status_code=400, detail="File must be an audio file (.wav, .mp3, .flac, .aiff)")
        
        result = await service.upload_timbre_file(file, sample_type)
        return result
    except Exception as e:
        logger.error(f"Error uploading timbre sample: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/timbre/mix")
async def set_timbre_mix(mix_request: TimbreMixRequest):
    """Set the timbre interpolation mix value"""
    try:
        result = await service.set_timbre_mix(mix_request.mix_value)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error setting timbre mix: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/timbre/status")
async def get_timbre_status():
    """Get current timbre interpolation status"""
    try:
        return service.get_timbre_status()
    except Exception as e:
        logger.error(f"Error getting timbre status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    """Serve audio files from the uploads directory"""
    try:
        file_path = os.path.join(service.upload_dir, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Audio file not found")
        
        # Determine media type based on file extension
        if filename.endswith('.wav'):
            media_type = 'audio/wav'
        elif filename.endswith('.mp3'):
            media_type = 'audio/mpeg'
        elif filename.endswith('.flac'):
            media_type = 'audio/flac'
        elif filename.endswith('.aiff'):
            media_type = 'audio/aiff'
        else:
            media_type = 'audio/wav'  # default
        
        return FileResponse(
            path=file_path,
            media_type=media_type,
            filename=filename
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving audio file {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/timbre/{sample_type}")
async def delete_timbre_sample(sample_type: str):
    """Delete a timbre sample file"""
    try:
        if sample_type not in ["sample_a", "sample_b"]:
            raise HTTPException(status_code=400, detail="sample_type must be 'sample_a' or 'sample_b'")
        
        if service.timbre_files[sample_type]:
            file_path = service.timbre_files[sample_type]["file_path"]
            if os.path.exists(file_path):
                os.remove(file_path)
            service.timbre_files[sample_type] = None
            logger.info(f"Deleted {sample_type}")
        
        return {"status": "deleted", "sample_type": sample_type}
    except Exception as e:
        logger.error(f"Error deleting timbre sample: {e}")
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
