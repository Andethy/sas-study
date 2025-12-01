#!/usr/bin/env python3
"""
Command-line interface for m1-timbre interpolation.
This module provides a headless interface to the timbre interpolation functionality.
"""

import argparse
import os
import sys
import torch
import torchaudio
import yaml
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from timbre.app import VAEInterp, resample, fit_to_block


class TimbreInterpolatorCLI:
    def __init__(self, config_path=None):
        """Initialize the CLI interpolator."""
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
        
        print(f"Timbre interpolator initialized with sample rate: {self.system_sr}")

    def load_audio(self, path):
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

    def interpolate(self, sample_a_path, sample_b_path, mix_ratio, output_path):
        """
        Perform timbre interpolation between two audio samples.
        
        Args:
            sample_a_path: Path to first audio sample
            sample_b_path: Path to second audio sample  
            mix_ratio: Interpolation ratio (0.0 = pure sample_b, 1.0 = pure sample_a)
            output_path: Path to save interpolated result
        """
        try:
            print(f"Loading audio files...")
            print(f"  Sample A: {sample_a_path}")
            print(f"  Sample B: {sample_b_path}")
            
            # Load audio files
            timbre_a = self.load_audio(sample_a_path)
            timbre_b = self.load_audio(sample_b_path)
            
            print(f"Performing interpolation with mix ratio: {mix_ratio}")
            
            # Perform interpolation
            with torch.no_grad():
                audio_interp = self.model(timbre_a, timbre_b, mix_ratio).detach()
                
                # Amplify the result (as done in original app)
                audio_interp *= 4
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save result
            torchaudio.save(output_path, audio_interp, sample_rate=self.system_sr)
            
            print(f"Interpolation complete. Output saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error during interpolation: {e}")
            return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Timbre Interpolation CLI")
    parser.add_argument("--sample-a", required=True, help="Path to first audio sample")
    parser.add_argument("--sample-b", required=True, help="Path to second audio sample")
    parser.add_argument("--mix", type=float, required=True, help="Mix ratio (0.0 to 1.0)")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--config", help="Path to config file (optional)")
    
    args = parser.parse_args()
    
    # Validate mix ratio
    if not 0.0 <= args.mix <= 1.0:
        print("Error: Mix ratio must be between 0.0 and 1.0")
        sys.exit(1)
    
    try:
        # Initialize interpolator
        interpolator = TimbreInterpolatorCLI(args.config)
        
        # Perform interpolation
        success = interpolator.interpolate(
            args.sample_a,
            args.sample_b, 
            args.mix,
            args.output
        )
        
        if success:
            print("SUCCESS: Timbre interpolation completed")
            sys.exit(0)
        else:
            print("ERROR: Timbre interpolation failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()