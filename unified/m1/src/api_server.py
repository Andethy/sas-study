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
import sys
import concurrent.futures

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
        self.realtime_player = None
        self._init_realtime_player()
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
    
    async def set_current_chord(self, group_id: str, symbol: str):
        """Immediately update the current chord"""
        async with self._lock:
            self.orch.set_current_chord(group_id, symbol)
        
        await self.manager.broadcast({
            "type": "chord_updated",
            "chord": symbol
        })
    
    async def queue_chord_by_tension(self, group_id: str, tension: float):
        """Queue chord for next measure based on tension value"""
        async with self._lock:
            # Get the harmonic instrument
            harm_inst = self.orch._harm_groups.get(group_id)
            if harm_inst and harm_inst.hg:
                # Only queue chord if tension is significant, otherwise let orchestrator use default
                if tension > 0.05:  # Only queue if tension > 5%
                    new_chord = harm_inst.hg.select_chord_by_tension(
                        current_tension=tension,
                        previous_chord=harm_inst._prev,
                        lambda_balance=1.0
                    )
                    # Queue it for next measure (replaces any existing queued chord)
                    harm_inst.queue_next_chord(new_chord)
                    
                    logger.info(f"Queued chord by tension {tension}: {new_chord} (prev: {harm_inst._prev})")
                    
                    await self.manager.broadcast({
                        "type": "chord_queued",
                        "chord": new_chord,
                        "tension": tension
                    })
                    
                    return new_chord
                else:
                    # Clear any queued chord to let orchestrator use default
                    harm_inst._queued_next = None
                    default_chord = f"{self.orch.cfg.key_root}_M"
                    
                    logger.info(f"Low tension {tension}, cleared queue for default chord: {default_chord}")
                    
                    await self.manager.broadcast({
                        "type": "chord_cleared",
                        "chord": default_chord,
                        "tension": tension
                    })
                    
                    return default_chord
            else:
                raise RuntimeError(f"Harmonic group {group_id} not found")
        
        return "C_M"  # fallback
    
    async def queue_custom_chord(self, group_id: str, notes: List[int]):
        """Queue a custom chord from MIDI notes"""
        async with self._lock:
            # Convert MIDI notes to chord symbol
            chord_symbol = self._notes_to_chord_symbol(notes)
            
            # Get the harmonic instrument and queue the chord
            harm_inst = self.orch._harm_groups.get(group_id)
            if harm_inst:
                harm_inst.queue_next_chord(chord_symbol)
                
                logger.info(f"Queued custom chord from notes {notes}: {chord_symbol}")
                
                await self.manager.broadcast({
                    "type": "chord_queued",
                    "chord": chord_symbol,
                    "notes": notes,
                    "custom": True
                })
                
                return chord_symbol
            else:
                raise RuntimeError(f"Harmonic group {group_id} not found")
    
    def _notes_to_chord_symbol(self, notes: List[int]) -> str:
        """Convert MIDI note numbers to chord symbol"""
        if len(notes) < 3:
            return "C_M"  # fallback
        
        # Sort notes and get the root (lowest note)
        sorted_notes = sorted(notes)
        root_note = sorted_notes[0] % 12
        
        # Map MIDI note numbers to note names
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        root_name = note_names[root_note]
        
        # Calculate intervals from root
        intervals = [(note - sorted_notes[0]) % 12 for note in sorted_notes[1:]]
        
        # Determine chord quality based on intervals
        if 4 in intervals and 7 in intervals:  # Major triad
            return f"{root_name}_M"
        elif 3 in intervals and 7 in intervals:  # Minor triad
            return f"{root_name}_m"
        elif 4 in intervals and 10 in intervals:  # Dominant 7th
            return f"{root_name}_7"
        elif 3 in intervals and 10 in intervals:  # Minor 7th
            return f"{root_name}_m7"
        elif 4 in intervals and 11 in intervals:  # Major 7th
            return f"{root_name}_M7"
        else:
            # Default to major if we can't identify
            return f"{root_name}_M"
    
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
    
    async def queue_fill_for_next_measure(self, preset: str, beats: int):
        async with self._lock:
            drums = self.orch._percussion.get("drums")
            if drums:
                # Queue fill for next measure instead of immediate trigger
                drums.queue_fill_beats(beats, preset=preset)
        
        await self.manager.broadcast({
            "type": "fill_queued",
            "preset": preset,
            "beats": beats
        })
    
    def _init_realtime_player(self):
        """Initialize the real-time timbre player."""
        try:
            # Add the m1-timbre src directory to Python path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            m1_timbre_src = os.path.join(repo_root, "safety", "m1-timbre", "src")
            
            if m1_timbre_src not in sys.path:
                sys.path.append(m1_timbre_src)
            
            # Import and initialize the real-time player
            from realtime_player import RealtimeTimbrePlayer
            
            self.realtime_player = RealtimeTimbrePlayer()
            self.realtime_player.set_status_callback(self._realtime_status_callback)
            
            logger.info("Real-time timbre player initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize real-time timbre player: {e}")
            self.realtime_player = None
    
    def _realtime_status_callback(self, message: str, status_type: str):
        """Callback for real-time player status updates."""
        logger.info(f"Realtime Player [{status_type}]: {message}")
        
        # Schedule broadcast in the main event loop (thread-safe)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule the coroutine to run in the main event loop
                asyncio.run_coroutine_threadsafe(
                    self.manager.broadcast({
                        "type": "realtime_status",
                        "message": message,
                        "status_type": status_type
                    }),
                    loop
                )
        except RuntimeError:
            # No event loop running, just log
            logger.warning(f"Could not broadcast realtime status: {message}")
    
    async def _check_and_prepare_realtime(self):
        """Check if both samples are uploaded and prepare real-time interpolations."""
        if not self.realtime_player:
            return
        
        if self.timbre_files["sample_a"] and self.timbre_files["sample_b"]:
            sample_a_path = self.timbre_files["sample_a"]["file_path"]
            sample_b_path = self.timbre_files["sample_b"]["file_path"]
            
            logger.info("Both samples uploaded, preparing real-time interpolations...")
            
            # Run preparation in thread pool to avoid blocking the event loop
            import concurrent.futures
            import threading
            
            def prepare_in_thread():
                try:
                    success = self.realtime_player.prepare_interpolations(sample_a_path, sample_b_path)
                    if success:
                        # Start playback automatically
                        self.realtime_player.start_playback()
                        return True
                    return False
                except Exception as e:
                    logger.error(f"Error in preparation thread: {e}")
                    return False
            
            # Use thread pool executor for better error handling
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = loop.run_in_executor(executor, prepare_in_thread)
                # Don't await here to avoid blocking the upload response
                asyncio.create_task(self._handle_preparation_result(future))
    
    async def _handle_preparation_result(self, future):
        """Handle the result of real-time preparation."""
        try:
            success = await future
            if success:
                logger.info("Real-time interpolation preparation completed successfully")
                await self.manager.broadcast({
                    "type": "realtime_ready",
                    "message": "Real-time timbre interpolation is ready",
                    "status": "success"
                })
            else:
                logger.error("Real-time interpolation preparation failed")
                await self.manager.broadcast({
                    "type": "realtime_error",
                    "message": "Failed to prepare real-time interpolation",
                    "status": "error"
                })
        except Exception as e:
            logger.error(f"Error handling preparation result: {e}")
            await self.manager.broadcast({
                "type": "realtime_error",
                "message": f"Preparation error: {str(e)}",
                "status": "error"
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
        
        # Check if both samples are uploaded and prepare real-time interpolations
        await self._check_and_prepare_realtime()
        
        return TimbreUploadResponse(
            filename=file.filename or "unknown",
            file_id=file_id,
            status="uploaded"
        )
    
    async def set_timbre_mix(self, mix_value: float):
        """Set the timbre interpolation mix value for real-time crossfading"""
        self.current_mix = max(0.0, min(1.0, mix_value))
        
        # Check if both files are uploaded
        if not (self.timbre_files["sample_a"] and self.timbre_files["sample_b"]):
            raise ValueError("Both sample A and sample B must be uploaded before mixing")
        
        # Use real-time player if available
        if self.realtime_player and self.realtime_player.is_prepared:
            try:
                # Update mix in real-time player (no regeneration needed!)
                self.realtime_player.set_mix(self.current_mix)
                
                logger.info(f"Real-time mix updated to: {self.current_mix}")
                
                # Broadcast update to connected clients
                await self.manager.broadcast({
                    "type": "timbre_mix_update",
                    "mix_value": self.current_mix,
                    "status": "realtime_update",
                    "is_realtime": True
                })
                
                return {
                    "status": "success", 
                    "mix_value": self.current_mix,
                    "is_realtime": True,
                    "message": "Real-time crossfade updated"
                }
                
            except Exception as e:
                logger.error(f"Real-time mix update error: {e}")
                raise RuntimeError(f"Real-time mix update failed: {str(e)}")
        
        else:
            # Fallback to old method if real-time player not available
            logger.warning("Real-time player not available, falling back to regeneration method")
            raise RuntimeError("Real-time player not initialized or not prepared")
    
    async def set_timbre_volume(self, volume: float):
        """Set the timbre playback volume"""
        volume = max(0.0, min(1.0, volume))
        
        if self.realtime_player:
            try:
                self.realtime_player.set_volume(volume)
                
                logger.info(f"Timbre volume updated to: {volume}")
                
                # Broadcast update to connected clients
                await self.manager.broadcast({
                    "type": "timbre_volume_update",
                    "volume": volume,
                    "status": "success"
                })
                
                return {
                    "status": "success",
                    "volume": volume,
                    "message": "Volume updated"
                }
                
            except Exception as e:
                logger.error(f"Volume update error: {e}")
                raise RuntimeError(f"Volume update failed: {str(e)}")
        else:
            raise RuntimeError("Real-time player not available")
    
    async def set_master_volume(self, volume: float):
        """Set master volume via OSC to port 10000"""
        volume = max(0.0, min(1.0, volume))
        
        try:
            # Send OSC message to master volume control
            import socket
            import struct
            
            # Create OSC message for "/rhythm" address
            address = "/rhythm"
            
            # OSC message format: address + type tag + value
            # Pad address to 4-byte boundary
            address_padded = address + '\0' * (4 - (len(address) % 4))
            
            # Type tag for float
            type_tag = ",f\0\0"
            
            # Pack float value
            value_packed = struct.pack('>f', volume)
            
            # Combine message
            osc_message = address_padded.encode() + type_tag.encode() + value_packed
            
            # Send to port 10000
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(osc_message, ('localhost', 10000))
            sock.close()
            
            logger.info(f"Master volume set to {volume} via OSC")
            
            # Broadcast update to connected clients
            await self.manager.broadcast({
                "type": "master_volume_update",
                "volume": volume,
                "status": "success"
            })
            
            return {
                "status": "success",
                "volume": volume,
                "message": f"Master volume set to {volume}"
            }
            
        except Exception as e:
            logger.error(f"Master volume update error: {e}")
            raise RuntimeError(f"Master volume update failed: {str(e)}")
    
    def get_timbre_status(self):
        """Get current timbre interpolation status"""
        realtime_status = self.realtime_player.get_status() if self.realtime_player else {}
        
        return {
            "sample_a": self.timbre_files["sample_a"]["filename"] if self.timbre_files["sample_a"] else None,
            "sample_b": self.timbre_files["sample_b"]["filename"] if self.timbre_files["sample_b"] else None,
            "current_mix": self.current_mix,
            "ready": bool(self.timbre_files["sample_a"] and self.timbre_files["sample_b"]),
            "realtime_status": realtime_status
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

class ChordTensionRequest(BaseModel):
    tension: float = Field(..., ge=0.0, le=1.0, description="Tension value for chord selection")
    group_id: str = Field(default="harm", description="Harmonic group ID")

@app.post("/chord/queue-by-tension")
async def queue_chord_by_tension(request: ChordTensionRequest):
    try:
        chord_symbol = await service.queue_chord_by_tension(request.group_id, request.tension)
        return {"status": "ok", "chord": chord_symbol}
    except Exception as e:
        logger.error(f"Error queuing chord by tension: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class CustomChordRequest(BaseModel):
    notes: List[int] = Field(..., description="Array of MIDI note numbers for custom chord")
    group_id: str = Field(default="harm", description="Harmonic group ID")

@app.post("/chord/queue-custom")
async def queue_custom_chord(request: CustomChordRequest):
    try:
        chord_symbol = await service.queue_custom_chord(request.group_id, request.notes)
        return {"status": "ok", "chord": chord_symbol}
    except Exception as e:
        logger.error(f"Error queuing custom chord: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fill")
async def trigger_fill(fill: DrumFill):
    try:
        await service.trigger_fill(fill.preset, fill.beats)
        return {"status": "ok", "preset": fill.preset, "beats": fill.beats}
    except Exception as e:
        logger.error(f"Error triggering fill: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fill/queue")
async def queue_fill_for_next_measure(fill: DrumFill):
    try:
        await service.queue_fill_for_next_measure(fill.preset, fill.beats)
        return {"status": "ok", "preset": fill.preset, "beats": fill.beats, "queued": True}
    except Exception as e:
        logger.error(f"Error queuing fill: {e}")
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

@app.post("/timbre/playback/start")
async def start_realtime_playback():
    """Start real-time timbre playback"""
    try:
        if not service.realtime_player:
            raise HTTPException(status_code=400, detail="Real-time player not initialized")
        
        success = service.realtime_player.start_playback()
        if success:
            return {"status": "success", "message": "Real-time playback started"}
        else:
            raise HTTPException(status_code=400, detail="Failed to start playback")
    except Exception as e:
        logger.error(f"Error starting real-time playback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/timbre/playback/stop")
async def stop_realtime_playback():
    """Stop real-time timbre playback"""
    try:
        if service.realtime_player:
            service.realtime_player.stop_playback()
        return {"status": "success", "message": "Real-time playback stopped"}
    except Exception as e:
        logger.error(f"Error stopping real-time playback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class TimbreVolumeRequest(BaseModel):
    volume: float = Field(..., ge=0.0, le=1.0, description="Volume level from 0.0 to 1.0")

@app.post("/timbre/volume")
async def set_timbre_volume(volume_request: TimbreVolumeRequest):
    """Set timbre playback volume"""
    try:
        result = await service.set_timbre_volume(volume_request.volume)
        return result
    except Exception as e:
        logger.error(f"Error setting timbre volume: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class MasterVolumeRequest(BaseModel):
    volume: float = Field(..., ge=0.0, le=1.0, description="Master volume level from 0.0 to 1.0")

@app.post("/master/volume")
async def set_master_volume(volume_request: MasterVolumeRequest):
    """Set master volume via OSC"""
    try:
        result = await service.set_master_volume(volume_request.volume)
        return result
    except Exception as e:
        logger.error(f"Error setting master volume: {e}")
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
