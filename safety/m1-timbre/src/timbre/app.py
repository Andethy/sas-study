import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
import numpy as np
import pygame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json

import os
import yaml
import librosa
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

from .timbreinterp.model.preprocessing import AudioSpec
from .timbreinterp.model.postprocessing import Vocoder
from .timbreinterp.model.model_builder import VanillaAE

from threading import Thread
import queue

drone_playing = False
def continuous_play_interp():
    while drone_playing:
        print("Drone Playing")

drone_thread = Thread(target=continuous_play_interp)
def play_drone():
    global drone_thread
    global drone_playing
    drone_playing = True
    print("Drone Playing")
    drone_thread = Thread(target=continuous_play_interp)
    drone_thread.start()

def stop_drone():
    global drone_thread
    global drone_playing
    drone_playing = False
    print("Stopped playing")
    drone_thread.join()

class App:

    def __init__(self):
        self.dir = os.path.dirname(os.path.abspath(__file__))
        print("FLAG:", os.path.abspath('../resources/results/average_ratings_normalized.csv'))
        self.data = pd.read_csv('../resources/results/average_ratings_normalized.csv')
        with open('../resources/results/TimbreResults.json') as f:
            self.timbre_data = json.load(f)

        self.safety_ratings = []
        self.urgency_ratings = []
        for key, value in self.timbre_data.items():
            self.safety_ratings.extend(value['Safety'])
            self.urgency_ratings.extend(value['Urgency'])
        
        self.init_ui()
        self.init_configs()
        self.init_model()

    
    def init_ui(self):
        self.app = tk.Tk()
        self.app.title("Sound Match Application")

        self.a_safety = tk.DoubleVar()
        self.a_urgency = tk.DoubleVar()
        self.b_safety = tk.DoubleVar()
        self.b_urgency = tk.DoubleVar()

        self.drone_button = ttk.Button(self.app, text="Drone", command=play_drone)
        self.drone_button.grid(row=0, column=0, pady=10)

        self.stop_drone_button = ttk.Button(self.app, text="Stop Drone", command=stop_drone)
        self.stop_drone_button.grid(row=0, column=1, pady=10)

        self.a_frame = ttk.LabelFrame(self.app, text="Sound A")
        self.a_frame.grid(row=1, column=0, padx=10, pady=10)

        tk.Label(self.a_frame, text="Safety").grid(row=0, column=0)
        self.a_safety_slider = ttk.Scale(self.a_frame, from_=-1, to=1, orient='horizontal', variable=self.a_safety)
        self.a_safety_slider.grid(row=0, column=1)

        tk.Label(self.a_frame, text="Urgency").grid(row=1, column=0)
        self.a_urgency_slider = ttk.Scale(self.a_frame, from_=-1, to=1, orient='horizontal', variable=self.a_urgency)
        self.a_urgency_slider.grid(row=1, column=1)

        self.b_frame = ttk.LabelFrame(self.app, text="Sound B")
        self.b_frame.grid(row=1, column=1, padx=10, pady=10)

        tk.Label(self.b_frame, text="Safety").grid(row=0, column=0)
        self.b_safety_slider = ttk.Scale(self.b_frame, from_=-1, to=1, orient='horizontal', variable=self.b_safety)
        self.b_safety_slider.grid(row=0, column=1)

        tk.Label(self.b_frame, text="Urgency").grid(row=1, column=0)
        self.b_urgency_slider = ttk.Scale(self.b_frame, from_=-1, to=1, orient='horizontal', variable=self.b_urgency)
        self.b_urgency_slider.grid(row=1, column=1)

        self.match_button = ttk.Button(self.app, text="Match Sounds", command=self.match_sounds)
        self.match_button.grid(row=2, column=0, columnspan=2, pady=10)

        self.a_audio_path = tk.StringVar()
        self.b_audio_path = tk.StringVar()

        self.result_label = ttk.Label(self.app, text="a: None\nb: None")
        self.result_label.grid(row=3, column=0, columnspan=2)

        self.mix = tk.DoubleVar()
        self.send_slider = ttk.Scale(self.app, from_=0, to=1, orient='horizontal', variable=self.mix)
        self.send_slider.grid(row=4, column=0, columnspan=2, pady=10)

        self.select_a_button = tk.Button(self.app, text="Select Sound A", command=lambda: self.open_file_dialog("A"))
        self.select_a_button.grid(row=5, column=0, pady=10)
        
        self.select_a_button = tk.Button(self.app, text="Select Sound B", command=lambda: self.open_file_dialog("B"))
        self.select_a_button.grid(row=5, column=1, pady=10)

        self.send_button = ttk.Button(self.app, text="Prepare Interpolation", command=lambda: self.prepare_interpolation(self.a_audio_path.get(), self.b_audio_path.get()))
        self.send_button.grid(row=6, column=0, pady=10)

        self.play_c_button = ttk.Button(self.app, text="Play Combined Sound", command=lambda: self.play_interp(self.mix.get()))
        self.play_c_button.grid(row=6, column=1, pady=10)

        self.play_a_button = ttk.Button(self.app, text="Play Sound A", command=self.play_a)
        self.play_a_button.grid(row=7, column=0, pady=10)

        self.play_b_button = ttk.Button(self.app, text="Play Sound B", command=self.play_b)
        self.play_b_button.grid(row=7, column=1, pady=10)

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 4))
        self.fig.suptitle('Safety and Urgency Ratings')

        self.ax1.boxplot(self.safety_ratings)
        self.ax1.set_title('Safety Ratings')
        self.ax1.set_ylabel('Rating')

        self.ax2.boxplot(self.urgency_ratings)
        self.ax2.set_title('Urgency Ratings')
        self.ax2.set_ylabel('Rating')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.app)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=8, column=0, columnspan=2, pady=10)

        self.app.protocol("WM_DELETE_WINDOW", self.on_close)
        pygame.mixer.init()
    
    def init_configs(self):
        config_path = f"{self.dir}/timbreinterp/config/VanillaVAE.yaml"
        with open(config_path) as f:
            self.configs = yaml.safe_load(f)
        self.system_sr = self.configs["audio"]["samplerate"]

    def init_model(self):
        self.model = VAEInterp(self.configs)
        checkpoint_path = "../resources/checkpoint/epoch=311-step=705120.ckpt"
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        self.model.eval()

    def run(self):
        self.app.mainloop()
    
    def on_close(self):
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
        print("Stopping audio")
        pygame.mixer.quit()
        print("Closing app")
        self.app.destroy()

    def find_closest_match(self, safety, urgency):
        distances = np.sqrt((self.data['stretched_safety'] - safety)**2 + (self.data['stretched_urgency'] - urgency)**2)
        return self.data.iloc[distances.idxmin()]

    @staticmethod
    def play_audio(file_path):
        try:
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            print('Sound played successfully?')
        except Exception as e:
            print(f"Error playing audio: {e}")

    def play_a(self):
        if self.a_audio_path:
            self.play_audio(self.a_audio_path.get())

    def play_b(self):
        if self.b_audio_path:
            self.play_audio(self.b_audio_path.get())

    def match_sounds(self):
        a_match = self.find_closest_match(self.a_safety.get(), self.a_urgency.get())
        b_match = self.find_closest_match(self.b_safety.get(), self.b_urgency.get())

        self.a_audio_path.set('../resources/samples/' + a_match['filepath'])
        self.b_audio_path.set('../resources/samples/' + b_match['filepath'])

        self.result_label.config(text=f"A: {self.a_audio_path.get()}\nB: {self.b_audio_path.get()}")

        a_safety_ratings = self.timbre_data[str(a_match['id'] + 3)]['Safety']
        a_urgency_ratings = self.timbre_data[str(a_match['id'] + 3)]['Urgency']

        b_safety_ratings = self.timbre_data[str(b_match['id'] + 3)]['Safety']
        b_urgency_ratings = self.timbre_data[str(b_match['id'] + 3)]['Urgency']

        self.ax1.clear()
        self.ax2.clear()

        self.ax1.boxplot([a_safety_ratings, a_urgency_ratings], labels=['Safety', 'Urgency'])
        self.ax1.set_title('a Sound Ratings')
        self.ax1.set_ylabel('Rating')
        self.ax1.set_ylim(-4, 4)

        self.ax2.boxplot([b_safety_ratings, b_urgency_ratings], labels=['Safety', 'Urgency'])
        self.ax2.set_title('b Sound Ratings')
        self.ax2.set_ylabel('Rating')
        self.ax2.set_ylim(-4, 4)

        self.canvas.draw()
    
    def open_file_dialog(self, side):

        if side == "A":
            file_path = filedialog.askopenfilename(title="Select a File",
                                               initialdir='../resources/samples',
                                               filetypes=[("Wav files", "*.wav"), ("All files", "*.*")])
            self.a_audio_path.set(file_path)
        
        elif side == "B":
            file_path = filedialog.askopenfilename(title="Select a File",
                                               initialdir='../resources/samples',
                                               filetypes=[("Wav files", "*.wav"), ("All files", "*.*")])
            self.b_audio_path.set(file_path)
        
        self.result_label.config(text=f"A: {self.a_audio_path.get()}\nB: {self.b_audio_path.get()}")
    
    def load_audio(self, path):
        audio, sr = torchaudio.load(path)
        if audio.shape[0] == 2:
            audio = audio.mean(0).unsqueeze(0)
        #audio = pitch_normalize(audio, sr)
        if sr != self.system_sr:
            audio = resample(audio, sr, self.system_sr)
        block_size = self.configs["audio"]["samplerate"] * self.configs["audio"]["default_len_in_s"]
        audio = fit_to_block(audio, block_size)
        return audio

    def prepare_interpolation(self, timbre_a_path: str, timbre_b_path: str):
        timbre1_audio = self.load_audio(timbre_a_path)
        timbre2_audio = self.load_audio(timbre_b_path)
        
        print(f"Generating interpolation points: 0.0")
        save_path0 = "../resources/interp_results/interp-0.0.wav"
        torchaudio.save(save_path0, timbre2_audio, sample_rate = self.system_sr)

        for i in range(1, 10):
            p = 0.1*i
            print(f"Generating interpolation points: {p:.1f}")
            audio_interp = self.model(timbre1_audio, timbre2_audio, p).detach()
            audio_interp *= 4
            save_path = f"../resources/interp_results/interp-{p:.1f}.wav"
            torchaudio.save(save_path, audio_interp, sample_rate = self.system_sr)
        
        print(f"Generating interpolation points: 1.0")
        save_path1 = "../resources/interp_results/interp-1.0.wav"
        torchaudio.save(save_path1, timbre1_audio, sample_rate = self.system_sr)
    
    def play_interp(self, p):
        p = round(1-p, 1)
        print(p)
        interp_audio_path = f"../resources/interp_results/interp-{p:.1f}.wav"
        self.play_audio(interp_audio_path)
    

class VAEInterp(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.preprocessing = AudioSpec(configs['audio'])
        self.model = VanillaAE(configs["model"])
        self.postprocessing = Vocoder(configs['audio'])

    def forward(self, timbre1, timbre2, r):

        timbre1 = timbre1.unsqueeze(0)
        timbre2 = timbre2.unsqueeze(0)
        spec1_real, spec1_imag = self.preprocessing(timbre1)
        spec2_real, spec2_imag = self.preprocessing(timbre2)
        
        emb1_real = self.model.encoder(spec1_real)
        emb1_imag = self.model.encoder(spec1_imag)
        emb2_real = self.model.encoder(spec2_real)
        emb2_imag = self.model.encoder(spec2_imag)
        
        emb_real_interp = emb1_real * r + emb2_real * (1 - r)
        emb_imag_interp = emb1_imag * r + emb2_imag * (1 - r)
        
        spec_real_interp = self.model.decoder(emb_real_interp)
        spec_imag_interp = self.model.decoder(emb_imag_interp)
        
        audio_interp = self.postprocessing(spec_real_interp, spec_imag_interp)
        audio_interp = audio_interp.squeeze(0)
        
        return audio_interp

def pitch_normalize(timbre_audio, sr):
    timbre_pitch = torchaudio.functional.detect_pitch_frequency(timbre_audio, sample_rate=sr)
    timbre_pitch, _ = timbre_pitch.mode()
    timbre_midi = round(librosa.hz_to_midi(timbre_pitch.item()))
    timbre_audio = torchaudio.functional.pitch_shift(timbre_audio, sample_rate=sr, n_steps=69-timbre_midi)
    return timbre_audio

def resample(audio, orig_sr, target_sr):
    resampled_waveform = torchaudio.functional.resample(
        audio,
        orig_sr,
        target_sr,
        lowpass_filter_width=64,
        rolloff=0.9475937167399596,
        resampling_method="sinc_interp_kaiser",
        beta=14.769656459379492,
    )   
    return resampled_waveform

def fit_to_block(timbre_audio, block_size):
    cur_len = timbre_audio.shape[1]
    if cur_len > block_size:
        timbre_audio = timbre_audio[..., :block_size]
    elif cur_len < block_size:
        padding = block_size - cur_len
        timbre_audio = F.pad(timbre_audio, (0, padding, 0, 0), "constant", 0)

    return timbre_audio


def main():
    app = App()
    app.run()

if __name__ == '__main__':
    main()
