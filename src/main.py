import socket
import tkinter as tk

def send_value(value):
    udp_ip = "127.0.0.1"
    udp_port = 54637  # Port number -> JUCE

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(value.encode(), (udp_ip, udp_port))

def on_slider_change(value):
    send_value(str(float(value) / 100))  # Send the slider value when it changes
    print("Changed to " + value)

def main():
    root = tk.Tk()
    root.title("Slider to UDP")
    root.geometry("400x50")

    slider = tk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL, command=on_slider_change)
    slider.pack()

    root.mainloop()

if __name__ == '__main__':
    main()