import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
import pygame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json


class App:
    def __init__(self):
        self.data = pd.read_csv('../../resources/timbre/results/average_ratings_normalized.csv')
        with open('../../resources/timbre/results/TimbreResults.json') as f:
            self.timbre_data = json.load(f)

        self.safety_ratings = []
        self.urgency_ratings = []
        for key, value in self.timbre_data.items():
            self.safety_ratings.extend(value['Safety'])
            self.urgency_ratings.extend(value['Urgency'])

        self.app = tk.Tk()
        self.app.title("Sound Match Application")

        self.a_safety = tk.DoubleVar()
        self.a_urgency = tk.DoubleVar()
        self.b_safety = tk.DoubleVar()
        self.b_urgency = tk.DoubleVar()

        self.a_frame = ttk.LabelFrame(self.app, text="Sound A")
        self.a_frame.grid(row=0, column=0, padx=10, pady=10)

        tk.Label(self.a_frame, text="Safety").grid(row=0, column=0)
        self.a_safety_slider = ttk.Scale(self.a_frame, from_=-1, to=1, orient='horizontal', variable=self.a_safety)
        self.a_safety_slider.grid(row=0, column=1)

        tk.Label(self.a_frame, text="Urgency").grid(row=1, column=0)
        self.a_urgency_slider = ttk.Scale(self.a_frame, from_=-1, to=1, orient='horizontal', variable=self.a_urgency)
        self.a_urgency_slider.grid(row=1, column=1)

        self.b_frame = ttk.LabelFrame(self.app, text="Sound B")
        self.b_frame.grid(row=0, column=1, padx=10, pady=10)

        tk.Label(self.b_frame, text="Safety").grid(row=0, column=0)
        self.b_safety_slider = ttk.Scale(self.b_frame, from_=-1, to=1, orient='horizontal', variable=self.b_safety)
        self.b_safety_slider.grid(row=0, column=1)

        tk.Label(self.b_frame, text="Urgency").grid(row=1, column=0)
        self.b_urgency_slider = ttk.Scale(self.b_frame, from_=-1, to=1, orient='horizontal', variable=self.b_urgency)
        self.b_urgency_slider.grid(row=1, column=1)

        self.a_audio_path = None
        self.b_audio_path = None
        self.output_audio_path = None

        self.result_label = ttk.Label(self.app, text="a: None\nb: None")
        self.result_label.grid(row=2, column=0, columnspan=2)

        self.mix = tk.DoubleVar()
        self.send_slider = ttk.Scale(self.app, from_=-1, to=1, orient='horizontal', variable=self.mix)
        self.send_slider.grid(row=3, column=0, columnspan=2, pady=10)

        self.send_button = ttk.Button(self.app, text="Send to Model", command=lambda: self.run_model(self.a_audio_path, self.b_audio_path, self.mix.get()))
        self.send_button.grid(row=4, column=0, pady=10)

        self.play_c_button = ttk.Button(self.app, text="Play Combined Sound", command=lambda: self.play_audio(self.output_audio_path))
        self.play_c_button.grid(row=4, column=1, pady=10)

        self.play_a_button = ttk.Button(self.app, text="Play Sound A", command=self.play_a)
        self.play_a_button.grid(row=5, column=0, pady=10)

        self.play_b_button = ttk.Button(self.app, text="Play Sound B", command=self.play_b)
        self.play_b_button.grid(row=5, column=1, pady=10)

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 4))
        self.fig.suptitle('Safety and Urgency Ratings')

        self.ax1.boxplot(self.safety_ratings)
        self.ax1.set_title('Safety Ratings')
        self.ax1.set_ylabel('Rating')

        self.ax2.boxplot(self.urgency_ratings)
        self.ax2.set_title('Urgency Ratings')
        self.ax2.set_ylabel('Rating')

        self.match_button = ttk.Button(self.app, text="Match Sounds", command=self.match_sounds)
        self.match_button.grid(row=1, column=0, columnspan=2, pady=10)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.app)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=6, column=0, columnspan=2, pady=10)

        pygame.mixer.init()

    def run(self):
        self.app.mainloop()

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
            self.play_audio(self.a_audio_path)

    def play_b(self):
        if self.b_audio_path:
            self.play_audio(self.b_audio_path)

    def match_sounds(self):
        a_match = self.find_closest_match(self.a_safety.get(), self.a_urgency.get())
        b_match = self.find_closest_match(self.b_safety.get(), self.b_urgency.get())

        self.a_audio_path = '../../resources/timbre/samples/' + a_match['filepath']
        self.b_audio_path = '../../resources/timbre/samples/' + b_match['filepath']

        self.result_label.config(text=f"A: {self.a_audio_path}\nB: {self.b_audio_path}")

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

    def run_model(self, sound_a: str, sound_b: str, mix: float):
        self.output_audio_path = run_model_callback(sound_a, sound_b, mix)

def run_model_callback(sound_a: str, sound_b: str, mix: float) -> str:
    """

    :param sound_a: file path to sound A
    :param sound_b: file path to sound B
    :param mix: float between -1 and 1, how much of sound A to mix with sound B
    :return: file path to the output audio
    """
    pass

if __name__ == '__main__':
    app = App()
    app.run()
