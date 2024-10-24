import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
import pygame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json

data = pd.read_csv('../../resources/timbre/results/average_ratings.csv')


with open('../../resources/timbre/results/TimbreResults.json') as f:
    timbre_data = json.load(f)

safety_ratings = []
urgency_ratings = []
for key, value in timbre_data.items():
    safety_ratings.extend(value['Safety'])
    urgency_ratings.extend(value['Urgency'])

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
app.title("Sound Match Application")

start_safety = tk.DoubleVar()
start_urgency = tk.DoubleVar()
end_safety = tk.DoubleVar()
end_urgency = tk.DoubleVar()

start_frame = ttk.LabelFrame(app, text="Start Sound")
start_frame.grid(row=0, column=0, padx=10, pady=10)

tk.Label(start_frame, text="Safety").grid(row=0, column=0)
start_safety_slider = ttk.Scale(start_frame, from_=-1, to=1, orient='horizontal', variable=start_safety)
start_safety_slider.grid(row=0, column=1)

tk.Label(start_frame, text="Urgency").grid(row=1, column=0)
start_urgency_slider = ttk.Scale(start_frame, from_=-1, to=1, orient='horizontal', variable=start_urgency)
start_urgency_slider.grid(row=1, column=1)

end_frame = ttk.LabelFrame(app, text="End Sound")
end_frame.grid(row=0, column=1, padx=10, pady=10)

tk.Label(end_frame, text="Safety").grid(row=0, column=0)
end_safety_slider = ttk.Scale(end_frame, from_=-1, to=1, orient='horizontal', variable=end_safety)
end_safety_slider.grid(row=0, column=1)

tk.Label(end_frame, text="Urgency").grid(row=1, column=0)
end_urgency_slider = ttk.Scale(end_frame, from_=-1, to=1, orient='horizontal', variable=end_urgency)
end_urgency_slider.grid(row=1, column=1)

start_audio_path = None
end_audio_path = None

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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle('Safety and Urgency Ratings')

ax1.boxplot(safety_ratings)
ax1.set_title('Safety Ratings')
ax1.set_ylabel('Rating')

ax2.boxplot(urgency_ratings)
ax2.set_title('Urgency Ratings')
ax2.set_ylabel('Rating')


def match_sounds():
    global start_audio_path, end_audio_path

    # Find the closest matches for start and end inputs
    start_match = find_closest_match(start_safety.get(), start_urgency.get())
    end_match = find_closest_match(end_safety.get(), end_urgency.get())

    # Update audio paths
    start_audio_path = '../../resources/timbre/samples/' + start_match['filepath']
    end_audio_path = '../../resources/timbre/samples/' + end_match['filepath']

    # Update result label
    result_label.config(text=f"Start: {start_audio_path}\nEnd: {end_audio_path}")

    # Get matched safety and urgency ratings for start and end
    start_safety_ratings = timbre_data[str(start_match['id'])]['Safety']
    start_urgency_ratings = timbre_data[str(start_match['id'])]['Urgency']

    end_safety_ratings = timbre_data[str(end_match['id'])]['Safety']
    end_urgency_ratings = timbre_data[str(end_match['id'])]['Urgency']

    # Clear previous plots
    ax1.clear()
    ax2.clear()

    # Update the left boxplot for start sound with two boxes (one for safety and one for urgency)
    ax1.boxplot([start_safety_ratings, start_urgency_ratings], labels=['Safety', 'Urgency'])
    ax1.set_title('Start Sound Ratings')
    ax1.set_ylabel('Rating')
    ax1.set_ylim(-4, 4)  # Set y-axis limits to ensure consistent scale

    # Update the right boxplot for end sound with two boxes (one for safety and one for urgency)
    ax2.boxplot([end_safety_ratings, end_urgency_ratings], labels=['Safety', 'Urgency'])
    ax2.set_title('End Sound Ratings')
    ax2.set_ylabel('Rating')
    ax2.set_ylim(-4, 4)  # Set y-axis limits to ensure consistent scale

    # Redraw canvas
    canvas.draw()


match_button = ttk.Button(app, text="Match Sounds", command=match_sounds)
match_button.grid(row=1, column=0, columnspan=2, pady=10)

canvas = FigureCanvasTkAgg(fig, master=app)
canvas.draw()
canvas.get_tk_widget().grid(row=4, column=0, columnspan=2, pady=10)

if __name__ == '__main__':
    app.mainloop()
