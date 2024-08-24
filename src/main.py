import socket
import time

def main():

    udp_ip = "127.0.0.1"
    udp_port = 54637  # Port number -> JUCE

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    while True:
        value = input("Update value: ")
        sock.sendto(value.encode(), (udp_ip, udp_port))
        time.sleep(0.1)  # Adjust the update speed (0.1 seconds)


if __name__ == '__main__':
    main()