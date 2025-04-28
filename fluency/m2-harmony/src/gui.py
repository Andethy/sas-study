import sys
import threading
import time
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from music21.instrument import SuspendedCymbal

import melody
import utils


class StaticPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("Curve Drawer")

        self.time_var = tk.DoubleVar(value=1.0)
        ttk.Label(root, text="Duration (seconds):").pack()
        self.time_slider = ttk.Scale(root, from_=0.1, to=10.0, variable=self.time_var, orient="horizontal")
        self.time_slider.pack(fill="x")

        self.divisions_var = tk.IntVar(value=10)
        ttk.Label(root, text="Number of Divisions:").pack()
        self.divisions_menu = ttk.Combobox(root, textvariable=self.divisions_var, values=[2, 4, 6, 8, 16])
        self.divisions_menu.pack()
        self.divisions_menu.current(1)

        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.canvas_widget = FigureCanvasTkAgg(self.fig, root)
        self.canvas_widget.get_tk_widget().pack(fill="both", expand=True)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_title("Draw Curve Here")

        self.x_data, self.y_data = [], []
        self.drawing = False

        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)

        # Play button
        self.play_button = ttk.Button(root, text="Play Curve", command=self.play_curve)
        self.play_button.pack()

        self.xarm = utils.StaticRobot("192.168.1.215", sim=True)


    def on_press(self, event):
        """Start a new curve when the mouse button is pressed."""
        if event.xdata is None or event.ydata is None:
            return
        self.x_data, self.y_data = [event.xdata], [event.ydata]
        self.drawing = True
        self.ax.clear()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.fig.canvas.draw()

    def on_motion(self, event):
        """Capture mouse movement only when the button is held down."""
        if not self.drawing or event.xdata is None or event.ydata is None:
            return
        self.x_data.append(event.xdata)
        self.y_data.append(event.ydata)
        self.ax.clear()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.plot(self.x_data, self.y_data, 'b-', lw=2)
        self.fig.canvas.draw()

    def on_release(self, event):
        """Stop drawing when the mouse button is released."""
        self.drawing = False

    def play_curve(self):
        if len(self.x_data) < 2:
            print("Draw a curve first!")
            return

        n = self.divisions_var.get()
        x_new = np.linspace(0, 1, n)
        y_new = np.interp(x_new, self.x_data, self.y_data)
        duration = self.time_var.get()
        interval_ms = int((duration / n) * 1000)  # Convert to milliseconds

        self.ax.clear()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.plot(self.x_data, self.y_data, 'b-', lw=2)  # Redraw base curve

        self.current_point = 0
        self.x_samples = x_new
        self.y_samples = y_new

        def animate():
            if self.current_point < len(self.x_samples):
                self.ax.clear()
                self.ax.set_xlim(0, 1)
                self.ax.set_ylim(0, 1)
                self.ax.plot(self.x_data, self.y_data, 'b-', lw=2)

                x = self.x_samples[self.current_point]
                y = self.y_samples[self.current_point]
                self.ax.plot(x, y, 'ro', markersize=8)
                self.fig.canvas.draw()

                self.current_point += 1
                self.root.after(interval_ms, animate)
            else:
                print("Playback complete.")

        animate()

        def run_robot():
            self.xarm.from_curve(self.y_samples.tolist(), duration)

        threading.Thread(target=run_robot, daemon=True).start()


class DynamicPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("Dynamic Tension Controller")
        self.root.geometry("200x150")

        # Tension slider
        self.tension_var = tk.DoubleVar(value=0.5)
        self.tension_slider = ttk.Scale(root, from_=0.0, to=1.0, variable=self.tension_var, orient="horizontal",
                                        command=self.on_tension_change)
        ttk.Label(root, text="Tension:").pack()
        self.tension_slider.pack(fill="x")

        # Preset ranges for tension
        self.count = 5
        self.tension_ranges = [0.0] + [i / self.count for i in range(1, self.count + 1)]
        self.current_chunk = self.get_tension_chunk(self.tension_var.get())

        # Robot instance
        self.xarm = utils.DynamicRobot("192.168.1.215", sim=True)

        # Toggle button
        self.robot_active = False
        self.toggle_button = ttk.Button(root, text="Turn ON", command=self.toggle_robot)
        self.toggle_button.pack(pady=10)

    def get_tension_chunk(self, value):
        """Determine which chunk the tension value falls into."""
        for i in range(len(self.tension_ranges) - 1):
            if self.tension_ranges[i] <= value < self.tension_ranges[i + 1]:
                return i
        return len(self.tension_ranges) - 1  # Return last chunk if at max value

    def on_tension_change(self, event):
        """Check if the slider entered a new chunk and update tension if needed."""
        if not self.robot_active:
            return  # Ignore changes if the robot is off

        new_chunk = self.get_tension_chunk(self.tension_var.get())
        if new_chunk != self.current_chunk:
            self.current_chunk = new_chunk
            # preset_tension = self.tension_ranges[new_chunk]
            print(f"Applying new tension preset: {new_chunk}")
            self.xarm.set_tension(self.current_chunk)

    def toggle_robot(self):
        """Toggle the robot's state between ON and OFF."""
        self.robot_active = not self.robot_active
        if self.robot_active:
            self.toggle_button.config(text="Turn OFF")
            print("Robot activated.")
            self.xarm.activate(self.tension_ranges)
        else:
            self.toggle_button.config(text="Turn ON")
            print("Robot deactivated.")
            self.xarm.deactivate()


class SuspendedPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("Suspended Tension Controller")
        self.root.geometry("300x500")

        # Create tension dropdown with 21 discrete options (0.0 to 1.0).
        self.tension_options = [round(i / 20.0, 2) for i in range(21)]
        self.tension_var = tk.StringVar(value=str(0.5))

        ttk.Label(root, text="Tension:").pack(pady=(10, 0))

        self.tension_dropdown = ttk.Combobox(
            root,
            textvariable=self.tension_var,
            values=[str(opt) for opt in self.tension_options],
            state="readonly"
        )
        self.tension_dropdown.pack(fill="x", padx=10, pady=5)
        self.tension_dropdown.bind("<<ComboboxSelected>>", self.on_tension_change)

        self.tension_frame = tk.Frame(root)
        self.tension_frame.pack(padx=10, pady=5)


        for idx, tension in enumerate(self.tension_options):
            btn = ttk.Button(
                self.tension_frame,
                text=str(tension),
                width=6,
                command=lambda t=tension: self.set_tension(t)
            )
            row = idx // 3
            col = idx % 3
            btn.grid(row=row, column=col, padx=2, pady=2)

        self.log_text = tk.Text(root, height=10, width=40)
        self.log_text.pack(padx=10, pady=5)
        self.log_text.insert(tk.END, "Log:\n")

        self.count = 5
        self.tension_ranges = [0.0] + [i / self.count for i in range(1, self.count + 1)]
        self.current_chunk = self.get_tension_chunk(float(self.tension_var.get()))

        self.xarm = utils.DynamicRobot("192.168.1.215", sim=True)
        self.robot_active = False
        self.toggle_button = ttk.Button(root, text="Turn ON", command=self.toggle_robot)
        self.toggle_button.pack(pady=10)

        self.hg = melody.HarmonyGenerator()
        self.prev_tension = 0
        self.prev_chord = self.xarm.set_chord_by_tension(0, 'C_M', self.hg, 0)
        self.log_text.insert(tk.END, f"Starting Chord: {self.prev_chord}\n")

    def get_tension_chunk(self, value):
        """Determine which chunk the tension value falls into."""
        for i in range(len(self.tension_ranges) - 1):
            if self.tension_ranges[i] <= value < self.tension_ranges[i + 1]:
                return i
        return len(self.tension_ranges) - 1

    def on_tension_change(self, event=None):
        try:
            current_tension = float(self.tension_var.get())
        except ValueError:
            current_tension = 0.5

        self.log_text.insert(tk.END, f"Tension changed to: {current_tension:.3f}\n")
        self.log_text.see(tk.END)

    def set_tension(self, tension):
        current_tension = float(tension)
        new_chunk = self.get_tension_chunk(current_tension)
        new_chord = None
        if new_chunk != self.current_chunk:
            self.current_chunk = new_chunk
        if self.robot_active:
            print(f"Applying new tension preset: {new_chunk}")
            # self.xarm.set_tension(new_chunk)
            new_chord = self.xarm.set_chord_by_tension(current_tension, self.prev_chord, self.hg, 0.1)

        progression_str = f"{self.prev_chord} -> {new_chord}"
        self.log_text.insert(tk.END, f"Chord progression updated: {progression_str}\n\n")
        self.log_text.see(tk.END)

        self.prev_chord = new_chord

    def toggle_robot(self):

        self.robot_active = not self.robot_active
        if self.robot_active:
            self.toggle_button.config(text="Turn OFF")
            print("Robot activated.")
            self.xarm.activate(self.tension_ranges)
        else:
            self.toggle_button.config(text="Turn ON")
            print("Robot deactivated.")
            self.xarm.deactivate()

class StaticPlayerV2:
    def __init__(self, root):
        self.root = root
        self.root.title("Static Tension Curve Player V2")

        # Duration slider
        self.time_var = tk.DoubleVar(value=1.0)
        ttk.Label(root, text="Duration (seconds):").pack()
        self.time_slider = ttk.Scale(root, from_=0.1, to=10.0, variable=self.time_var, orient="horizontal")
        self.time_slider.pack(fill="x")

        # Number of divisions
        self.divisions_var = tk.IntVar(value=10)
        ttk.Label(root, text="Number of Divisions:").pack()
        self.divisions_menu = ttk.Combobox(root, textvariable=self.divisions_var, values=[2, 4, 6, 8, 16])
        self.divisions_menu.pack()
        self.divisions_menu.current(1)

        self.backend_curves = {
            "Linear": lambda x: x,
            "Quad": lambda x: x**2,
            "Concave": lambda x: -4 * x ** 2 + 4 * x,
            "Sine Wave": lambda x: 0.5 * (np.sin(2 * np.pi * x) + 1),
            "Step": lambda x: (x > 0.5).astype(float)
        }

        self.selected_curve = tk.StringVar(value="Linear")
        ttk.Label(root, text="Backend Curve:").pack()
        self.curve_selector = ttk.Combobox(root, textvariable=self.selected_curve, values=list(self.backend_curves.keys()))
        self.curve_selector.pack()

        # Canvas for drawing
        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.canvas_widget = FigureCanvasTkAgg(self.fig, root)
        self.canvas_widget.get_tk_widget().pack(fill="both", expand=True)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_title("Draw your curve after sound playback")

        self.x_data = []
        self.y_data = []
        self.drawing = False
        self.draw_enabled = False

        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)

        self.play_sound_button = ttk.Button(root, text="Play Sound", command=self.play_sound)
        self.play_sound_button.pack(pady=5)
        self.show_curve_button = ttk.Button(root, text="Show Curve", command=self.show_curve)
        self.show_curve_button.pack(pady=5)

        self.xarm = utils.StaticRobot("192.168.1.215", sim=True)

    def on_press(self, event):
        if not self.draw_enabled:
            return
        if event.xdata is None or event.ydata is None:
            return
        self.x_data = [event.xdata]
        self.y_data = [event.ydata]
        self.drawing = True
        if hasattr(self, 'draw_line') and self.draw_line:
            self.draw_line.remove()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.fig.canvas.draw()

    def on_motion(self, event):
        if not self.draw_enabled or not self.drawing or event.xdata is None or event.ydata is None:
            return
        self.x_data.append(event.xdata)
        self.y_data.append(event.ydata)
        if hasattr(self, 'draw_line') and self.draw_line:
            self.draw_line.remove()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.plot(self.x_data, self.y_data, 'r-', lw=2)  # Free drawn curve
        self.fig.canvas.draw()

    def on_release(self, event):
        if self.draw_enabled:
            self.drawing = False

    def play_sound(self):
        self.ax.clear()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        duration = self.time_var.get()
        n = self.divisions_var.get()
        x_samples = np.linspace(0, 1, n)  # Red dot X-positions

        backend_fn = self.backend_curves[self.selected_curve.get()]
        y_samples_to_send = backend_fn(np.linspace(0, 1, n))

        # Animate red dot while sound plays
        self.current_point = 0
        interval_ms = int((duration / n) * 1000)

        def animate():
            if self.current_point < n:
                if hasattr(self, 'red_dot'):
                    self.red_dot.remove()

                x = x_samples[self.current_point]
                self.red_dot, = self.ax.plot(x, 0, 'ro', markersize=8)
                self.fig.canvas.draw()
                self.current_point += 1
                self.root.after(interval_ms, animate)
            else:
                print("Playback complete.")

        animate()

        def run_robot():
            self.xarm.from_curve(y_samples_to_send.tolist(), duration)

        threading.Thread(target=run_robot, daemon=True).start()

        # Enable drawing
        self.draw_enabled = True
        self.x_data = []
        self.y_data = []

    def show_curve(self):
        # Reveal the real evaluated curve based on the user's drawn domain
        if len(self.x_data) < 2:
            print("Please draw a curve first!")
            return

        max_x = max(self.x_data)
        n = self.divisions_var.get()
        x_domain = np.linspace(0, max_x, n)
        x_normalized = x_domain / max_x if max_x != 0 else x_domain
        backend_fn = self.backend_curves[self.selected_curve.get()]
        y_new = backend_fn(x_normalized)

        self.x_samples = x_normalized
        self.y_samples = y_new

        self.ax.clear()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.plot(self.x_data, self.y_data, 'r--', lw=1, label="User Curve")
        self.ax.plot(x_normalized, y_new, 'b-', lw=2, label="Actual Curve")
        self.ax.legend()
        self.fig.canvas.draw()

    def play_curve(self):
        if not hasattr(self, 'y_samples') or len(self.y_samples) == 0:
            print("Run 'Show Curve' first!")
            return

        duration = self.time_var.get()
        n = len(self.y_samples)
        interval_ms = int((duration / n) * 1000)

        self.current_point = 0

        def animate():
            if self.current_point < n:
                self.ax.clear()
                self.ax.set_xlim(0, 1)
                self.ax.set_ylim(0, 1)
                self.ax.plot(self.x_samples, self.y_samples, 'b-', lw=2)
                x = self.x_samples[self.current_point]
                y = self.y_samples[self.current_point]
                self.ax.plot(x, y, 'ro', markersize=8)
                self.fig.canvas.draw()
                self.current_point += 1
                self.root.after(interval_ms, animate)
            else:
                print("Playback complete.")

        animate()

        def run_robot():
            self.xarm.from_curve(self.y_samples.tolist(), duration)

        threading.Thread(target=run_robot, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    if sys.argv[1] == 'static':
        StaticPlayerV2(root)
    else:
        SuspendedPlayer(root)
    root.mainloop()

