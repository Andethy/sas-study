#!/usr/bin/env python3
"""
Real-time crossfading audio player for timbre interpolation.
Pre-generates interpolation points and provides smooth crossfading between them.
"""

import os
import sys
import threading
import time
import numpy as np
import pygame
import torch
import torchaudio
from typing import Optional, List, Callable
import yaml

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from timbre.app import VAEInterp, resample, fit_to_block


class RealtimeTimbrePlayer:
    def __init__(self, config_path=None):
        """Initialize the real-time timbre player."""
        self.dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load configuration
        if config_path is None:
            config_path = os.path.join(self.dir, "timbre/timbreinterp/config/VanillaVAE.yaml")
        
        with open(config_path) as f:
            self.configs = yaml.safe_load(f)
        
        self.system_sr = self.configs["audio"]["samplerate"]
        
        # Initialize model
        self.model = VAEInterp(self.configs)
        
        # Load checkpoint
        checkpoint_path = os.path.join(self.dir, "../resources/checkpoint/epoch=311-step=705120.ckpt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        self.model.eval()
        
        # Initialize pygame mixer
        try:
            pygame.mixer.pre_init(frequency=self.system_sr, size=-16, channels=1, buffer=1024)
            pygame.mixer.init()
            self._log_status(f"Audio system initialized at {self.system_sr}Hz", "info")
        except Exception as e:
            self._log_status(f"Warning: Audio system initialization failed: {e}", "error")
            # Continue without audio for now
        
        # State variables
        self.interpolation_points: List[np.ndarray] = []
        self.current_mix = 0.5
        self.is_playing = False
        self.is_prepared = False
        self.play_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Crossfading parameters
        self.crossfade_duration = 0.1  # 100ms crossfade
        self.loop_duration = 2.0  # 2 second loops
        self.crossfade_start_time = 1.0  # Start crossfading 1 second into the loop
        
        # Playback state
        self.current_playing_mix = 0.5  # The mix ratio currently being played
        self.target_mix = 0.5  # The target mix ratio (from slider)
        self.volume = 1.0  # Volume level (0.0 to 1.0)
        
        # Advanced crossfading parameters
        self.chunk_overlap_ratio = 0.5  # 50% overlap between chunks
        self.fade_duration = 0.1  # 100ms fade in/out for each chunk
        
        # Callback for status updates
        self.status_callback: Optional[Callable[[str, str], None]] = None
        
        print(f"Real-time timbre player initialized with sample rate: {self.system_sr}")

    def set_status_callback(self, callback: Callable[[str, str], None]):
        """Set callback function for status updates (message, type)."""
        self.status_callback = callback

    def _log_status(self, message: str, status_type: str = "info"):
        """Log status message via callback if available."""
        try:
            if self.status_callback:
                self.status_callback(message, status_type)
            else:
                print(f"[{status_type.upper()}] {message}")
        except Exception as e:
            # Fallback to print if callback fails
            print(f"[{status_type.upper()}] {message}")
            print(f"[WARNING] Status callback failed: {e}")

    def load_audio(self, path: str) -> torch.Tensor:
        """Load and preprocess audio file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio file not found: {path}")
        
        audio, sr = torchaudio.load(path)
        
        # Convert stereo to mono if needed
        if audio.shape[0] == 2:
            audio = audio.mean(0).unsqueeze(0)
        
        # Resample if needed
        if sr != self.system_sr:
            audio = resample(audio, sr, self.system_sr)
        
        # Fit to expected block size
        block_size = self.configs["audio"]["samplerate"] * self.configs["audio"]["default_len_in_s"]
        audio = fit_to_block(audio, block_size)
        
        return audio

    def prepare_interpolations(self, sample_a_path: str, sample_b_path: str) -> bool:
        """
        Pre-generate all interpolation points for real-time crossfading.
        
        Args:
            sample_a_path: Path to first audio sample
            sample_b_path: Path to second audio sample
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self._log_status("Loading audio samples...", "info")
            
            # Load audio files
            timbre_a = self.load_audio(sample_a_path)
            timbre_b = self.load_audio(sample_b_path)
            
            self._log_status("Generating interpolation points...", "info")
            
            # Generate 11 interpolation points (0.0, 0.1, 0.2, ..., 1.0)
            self.interpolation_points = []
            
            for i in range(11):
                mix_ratio = i / 10.0
                self._log_status(f"Generating interpolation point {i+1}/11 (mix: {mix_ratio:.1f})", "info")
                
                with torch.no_grad():
                    # Note: mix_ratio here is already in the correct format for the model
                    # 0.0 = pure sample_b, 1.0 = pure sample_a
                    audio_interp = self.model(timbre_a, timbre_b, mix_ratio).detach()
                    
                    # Amplify and convert to numpy
                    audio_interp *= 4
                    audio_np = audio_interp.squeeze().numpy()
                    
                    # Normalize to prevent clipping
                    max_val = np.max(np.abs(audio_np))
                    if max_val > 0:
                        audio_np = audio_np / max_val * 0.8  # Leave some headroom
                    
                    # Convert to 16-bit integer for pygame
                    audio_int16 = (audio_np * 32767).astype(np.int16)
                    
                    self.interpolation_points.append(audio_int16)
            
            self.is_prepared = True
            self._log_status("Interpolation points generated successfully!", "success")
            return True
            
        except Exception as e:
            self._log_status(f"Error preparing interpolations: {e}", "error")
            return False

    def set_mix(self, mix_value: float):
        """
        Set the target mix value for real-time crossfading.
        
        Args:
            mix_value: Mix ratio from 0.0 (pure A) to 1.0 (pure B)
        """
        self.target_mix = max(0.0, min(1.0, mix_value))
        self.current_mix = self.target_mix  # Also update current for immediate response

    def set_volume(self, volume: float):
        """
        Set the playback volume.
        
        Args:
            volume: Volume level from 0.0 (silent) to 1.0 (full volume)
        """
        self.volume = max(0.0, min(1.0, volume))

    def _crossfade_audio(self, audio1: np.ndarray, audio2: np.ndarray, fade_ratio: float) -> np.ndarray:
        """
        Crossfade between two audio arrays.
        
        Args:
            audio1: First audio array
            audio2: Second audio array
            fade_ratio: Crossfade ratio (0.0 = pure audio1, 1.0 = pure audio2)
            
        Returns:
            Crossfaded audio array
        """
        min_len = min(len(audio1), len(audio2))
        audio1_trimmed = audio1[:min_len]
        audio2_trimmed = audio2[:min_len]
        
        return (audio1_trimmed * (1.0 - fade_ratio) + audio2_trimmed * fade_ratio).astype(np.int16)
    
    def _apply_fade_envelope(self, audio: np.ndarray, fade_in_samples: int, fade_out_samples: int) -> np.ndarray:
        """
        Apply fade-in and fade-out envelopes to audio to prevent clicks and pops.
        
        Args:
            audio: Input audio array
            fade_in_samples: Number of samples for fade-in
            fade_out_samples: Number of samples for fade-out
            
        Returns:
            Audio with fade envelopes applied
        """
        if len(audio) == 0:
            return audio
        
        audio_float = audio.astype(np.float32)
        
        # Apply fade-in
        if fade_in_samples > 0 and len(audio) > fade_in_samples:
            fade_in_curve = np.linspace(0.0, 1.0, fade_in_samples)
            audio_float[:fade_in_samples] *= fade_in_curve
        
        # Apply fade-out
        if fade_out_samples > 0 and len(audio) > fade_out_samples:
            fade_out_curve = np.linspace(1.0, 0.0, fade_out_samples)
            audio_float[-fade_out_samples:] *= fade_out_curve
        
        return audio_float.astype(np.int16)

    def _get_current_audio(self) -> np.ndarray:
        """
        Get the current audio based on the current mix value.
        
        Returns:
            Current audio array for playback
        """
        return self._get_audio_for_mix(self.current_mix)
    
    def _get_audio_for_mix(self, mix_value: float) -> np.ndarray:
        """
        Get audio for a specific mix value with crossfading between interpolation points.
        
        Args:
            mix_value: Mix ratio from 0.0 (pure A) to 1.0 (pure B)
            
        Returns:
            Audio array for the specified mix
        """
        if not self.is_prepared or len(self.interpolation_points) == 0:
            return np.zeros(1024, dtype=np.int16)
        
        # Map mix value (0.0 to 1.0) to interpolation point indices
        # UI: 0% B = pure A, 100% B = pure B
        # Interpolation points: index 0 = pure B, index 10 = pure A
        # So UI 0% B should map to interpolation point index 10 (pure A)
        # And UI 100% B should map to interpolation point index 0 (pure B)
        inverted_mix = 1.0 - mix_value
        scaled_mix = inverted_mix * (len(self.interpolation_points) - 1)
        
        # Get the two adjacent interpolation points
        lower_idx = int(np.floor(scaled_mix))
        upper_idx = min(lower_idx + 1, len(self.interpolation_points) - 1)
        
        # Calculate crossfade ratio between the two points
        fade_ratio = scaled_mix - lower_idx
        
        if lower_idx == upper_idx:
            # Exact match to an interpolation point
            return self.interpolation_points[lower_idx]
        else:
            # Crossfade between two interpolation points
            return self._crossfade_audio(
                self.interpolation_points[lower_idx],
                self.interpolation_points[upper_idx],
                fade_ratio
            )

    def _play_loop(self):
        """Main playback loop with smooth overlapping crossfades."""
        self._log_status("Starting real-time crossfading playback loop", "info")
        
        # Check if pygame mixer is available
        if not pygame.mixer.get_init():
            self._log_status("Pygame mixer not initialized, cannot start playback", "error")
            return
        
        # Initialize playback state
        self.current_playing_mix = self.current_mix
        loop_start_time = time.time()
        last_chunk_time = 0
        
        # Chunk parameters for smooth overlapping playback
        chunk_duration = 1.0  # 1 second chunks
        overlap_duration = 0.5  # 500ms overlap
        chunk_interval = chunk_duration - overlap_duration  # 500ms between chunk starts
        
        while not self.stop_event.is_set():
            try:
                current_time = time.time()
                
                # Check if it's time to start a new chunk
                if current_time - last_chunk_time >= chunk_interval:
                    loop_elapsed = (current_time - loop_start_time) % self.loop_duration
                    
                    # Determine what mix to play based on loop position and target
                    if loop_elapsed < self.crossfade_start_time:
                        # First part of loop: play current mix
                        playback_mix = self.current_playing_mix
                    else:
                        # Second part of loop: crossfade to target mix
                        crossfade_progress = (loop_elapsed - self.crossfade_start_time) / (self.loop_duration - self.crossfade_start_time)
                        crossfade_progress = min(1.0, crossfade_progress)
                        
                        # Smooth crossfade from current to target
                        playback_mix = self.current_playing_mix * (1.0 - crossfade_progress) + self.target_mix * crossfade_progress
                    
                    # Get audio for the calculated mix
                    audio_data = self._get_audio_for_mix(playback_mix)
                    
                    if len(audio_data) > 0:
                        # Create 1-second chunk with proper fade envelopes
                        chunk_samples = int(self.system_sr * chunk_duration)
                        if len(audio_data) > chunk_samples:
                            audio_chunk = audio_data[:chunk_samples]
                        else:
                            # Loop the audio if it's shorter than chunk duration
                            repeats_needed = int(np.ceil(chunk_samples / len(audio_data)))
                            audio_chunk = np.tile(audio_data, repeats_needed)[:chunk_samples]
                        
                        # Apply smooth fade envelopes for overlapping
                        fade_samples = int(self.system_sr * overlap_duration)  # 500ms fade in/out
                        audio_chunk = self._apply_fade_envelope(audio_chunk, fade_samples, fade_samples)
                        
                        # Apply volume control
                        if self.volume != 1.0:
                            audio_chunk = (audio_chunk.astype(np.float32) * self.volume).astype(np.int16)
                        
                        # Play the chunk (non-blocking)
                        try:
                            sound = pygame.sndarray.make_sound(audio_chunk)
                            sound.play()  # Don't wait for completion - allows overlapping
                        except Exception as e:
                            self._log_status(f"Error playing audio chunk: {e}", "error")
                    
                    last_chunk_time = current_time
                    
                    # Check if we've completed a loop
                    if loop_elapsed >= self.loop_duration - chunk_interval:
                        # Update current playing mix to target for next loop
                        self.current_playing_mix = self.target_mix
                        loop_start_time = current_time
                
                # Small sleep to prevent busy waiting
                time.sleep(0.01)
                
            except Exception as e:
                self._log_status(f"Error in playback loop: {e}", "error")
                time.sleep(0.1)  # Brief pause before retrying

    def start_playback(self) -> bool:
        """
        Start continuous playback with real-time crossfading.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        if not self.is_prepared:
            self._log_status("Cannot start playback: interpolations not prepared", "error")
            return False
        
        if self.is_playing:
            self._log_status("Playback already running", "info")
            return True
        
        try:
            self.stop_event.clear()
            self.play_thread = threading.Thread(target=self._play_loop, daemon=True)
            self.play_thread.start()
            self.is_playing = True
            self._log_status("Real-time playback started", "success")
            return True
            
        except Exception as e:
            self._log_status(f"Error starting playback: {e}", "error")
            return False

    def stop_playback(self):
        """Stop continuous playback."""
        if not self.is_playing:
            return
        
        self._log_status("Stopping real-time playback", "info")
        self.stop_event.set()
        
        if self.play_thread and self.play_thread.is_alive():
            self.play_thread.join(timeout=2.0)
        
        pygame.mixer.stop()
        self.is_playing = False
        self._log_status("Real-time playback stopped", "info")

    def cleanup(self):
        """Clean up resources."""
        self.stop_playback()
        pygame.mixer.quit()

    def get_status(self) -> dict:
        """Get current player status."""
        return {
            "is_prepared": self.is_prepared,
            "is_playing": self.is_playing,
            "current_mix": self.current_mix,
            "interpolation_points": len(self.interpolation_points)
        }


def main():
    """CLI entry point for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time Timbre Player")
    parser.add_argument("--sample-a", required=True, help="Path to first audio sample")
    parser.add_argument("--sample-b", required=True, help="Path to second audio sample")
    parser.add_argument("--config", help="Path to config file (optional)")
    
    args = parser.parse_args()
    
    try:
        # Initialize player
        player = RealtimeTimbrePlayer(args.config)
        
        # Prepare interpolations
        if not player.prepare_interpolations(args.sample_a, args.sample_b):
            print("Failed to prepare interpolations")
            sys.exit(1)
        
        # Start playback
        if not player.start_playback():
            print("Failed to start playback")
            sys.exit(1)
        
        print("Real-time playback started. Press Enter to stop...")
        input()
        
        player.cleanup()
        print("Playback stopped.")
        
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()