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
        """Sample the curve and call the callback."""
        if len(self.x_data) < 2:
            print("Draw a curve first!")
            return

        # Interpolating to get n points
        n = self.divisions_var.get()
        x_new = np.linspace(0, 1, n)
        y_new = np.interp(x_new, self.x_data, self.y_data)

        duration = self.time_var.get()
        print(f"Playing curve over {duration} seconds with {n} samples:")
        print(y_new.tolist())

        # Replace this with your actual callback function
        self.xarm.from_curve(y_new.tolist(), duration)


if __name__ == "__main__":
    root = tk.Tk()
    app = StaticPlayer(root)
    root.mainloop()