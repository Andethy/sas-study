import tkinter as tk
from pythonosc import udp_client

# OSC Configuration
ip = "127.0.0.1"
port = 54637

client: udp_client.SimpleUDPClient

# Function to handle slider value change and send it via OSC
def send_value(value):
    x = float(value)
    client.send_message("/melody", x)  # Send the slider value to the OSC route
    print("Sent", x)

# Main application
def main():

    client = udp_client.SimpleUDPClient(ip, port)

    # Create the Tkinter window
    root = tk.Tk()
    root.title("OSC Slider")

    # Create a label
    label = tk.Label(root, text="Melody Slider")
    label.pack()

    # Create the slider widget
    slider = tk.Scale(root, from_=0, to=1, orient="horizontal", resolution=0.01, command=send_value)
    slider.pack()

    # Start the Tkinter main loop
    root.mainloop()

if __name__ == '__main__':
    main()
