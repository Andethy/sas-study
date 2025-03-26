import threading
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import utils


class StaticPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("Curve Drawer")

        # Float slider for duration
        self.time_var = tk.DoubleVar(value=1.0)
        ttk.Label(root, text="Duration (seconds):").pack()
        self.time_slider = ttk.Scale(root, from_=0.1, to=10.0, variable=self.time_var, orient="horizontal")
        self.time_slider.pack(fill="x")

        # Number of divisions dropdown
        self.divisions_var = tk.IntVar(value=10)
        ttk.Label(root, text="Number of Divisions:").pack()
        self.divisions_menu = ttk.Combobox(root, textvariable=self.divisions_var, values=[2, 4, 6, 8, 16])
        self.divisions_menu.pack()
        self.divisions_menu.current(1)

        # Canvas for drawing
        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.canvas_widget = FigureCanvasTkAgg(self.fig, root)
        self.canvas_widget.get_tk_widget().pack(fill="both", expand=True)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_title("Draw Curve Here")

        # Mouse interaction
        self.x_data, self.y_data = [], []
        self.drawing = False  # Track whether the mouse button is held

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

        # State for animation
        self.current_point = 0
        self.x_samples = x_new
        self.y_samples = y_new

        def animate():
            if self.current_point < len(self.x_samples):
                # Redraw base curve
                self.ax.clear()
                self.ax.set_xlim(0, 1)
                self.ax.set_ylim(0, 1)
                self.ax.plot(self.x_data, self.y_data, 'b-', lw=2)

                # Draw red dot at current point
                x = self.x_samples[self.current_point]
                y = self.y_samples[self.current_point]
                self.ax.plot(x, y, 'ro', markersize=8)
                self.fig.canvas.draw()

                self.current_point += 1
                self.root.after(interval_ms, animate)
            else:
                print("Playback complete.")

        animate()

        # Run the robot command in a background thread
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

if __name__ == "__main__":
    root = tk.Tk()
    app = StaticPlayer(root)
    root.mainloop()
