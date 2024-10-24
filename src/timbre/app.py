import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
import pygame

data = pd.read_csv('../../resources/timbre/results/average_ratings.csv')
pygame.mixer.init()

def find_closest_match(safety, urgency):
    distances = np.sqrt((data['average safety'] - safety)**2 + (data['average urgency'] - urgency)**2)
    return data.iloc[distances.idxmin()]


def play_audio(file_path):
    try:
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
    except Exception as e:
        print(f"Error playing audio: {e}")


app = tk.Tk()
app.title("Timbre Interpolation Match")

start_safety = tk.DoubleVar()
start_urgency = tk.DoubleVar()
end_safety = tk.DoubleVar()
end_urgency = tk.DoubleVar()

start_frame = ttk.LabelFrame(app, text="Start Sound")
start_frame.grid(row=0, column=0, padx=10, pady=10)

tk.Label(start_frame, text="Safety").grid(row=0, column=0)
start_safety_slider = ttk.Scale(start_frame, from_=-2.5, to=2.5, orient='horizontal', variable=start_safety)
start_safety_slider.grid(row=0, column=1)

tk.Label(start_frame, text="Urgency").grid(row=1, column=0)
start_urgency_slider = ttk.Scale(start_frame, from_=-2.5, to=2.5, orient='horizontal', variable=start_urgency)
start_urgency_slider.grid(row=1, column=1)

end_frame = ttk.LabelFrame(app, text="End Sound")
end_frame.grid(row=0, column=1, padx=10, pady=10)

tk.Label(end_frame, text="Safety").grid(row=0, column=0)
end_safety_slider = ttk.Scale(end_frame, from_=-2.5, to=2.5, orient='horizontal', variable=end_safety)
end_safety_slider.grid(row=0, column=1)

tk.Label(end_frame, text="Urgency").grid(row=1, column=0)
end_urgency_slider = ttk.Scale(end_frame, from_=-2.5, to=2.5, orient='horizontal', variable=end_urgency)
end_urgency_slider.grid(row=1, column=1)

start_audio_path = None
end_audio_path = None

def match_sounds():
    global start_audio_path, end_audio_path
    start_match = find_closest_match(start_safety.get(), start_urgency.get())
    end_match = find_closest_match(end_safety.get(), end_urgency.get())
    start_audio_path = '../../resources/timbre/samples/' + start_match['filepath']
    end_audio_path = '../../resources/timbre/samples/' + end_match['filepath']
    result_label.config(text=f"Start: {start_audio_path}\nEnd: {end_audio_path}")

match_button = ttk.Button(app, text="Match Sounds", command=match_sounds)
match_button.grid(row=1, column=0, columnspan=2, pady=10)

result_label = ttk.Label(app, text="Start: None\nEnd: None")
result_label.grid(row=2, column=0, columnspan=2)

def play_start():
    if start_audio_path:
        play_audio(start_audio_path)

def play_end():
    if end_audio_path:
        play_audio(end_audio_path)

play_start_button = ttk.Button(app, text="Play Start Sound", command=play_start)
play_start_button.grid(row=3, column=0, pady=10)

play_end_button = ttk.Button(app, text="Play End Sound", command=play_end)
play_end_button.grid(row=3, column=1, pady=10)

if __name__ == '__main__':
    app.mainloop()