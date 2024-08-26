import socket
import time

def main():

    udp_ip = "127.0.0.1"
    udp_port = 54637  # Port number -> JUCE

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    value = 'tmp'
    while value:
        value = input("Update value: ")
        sock.sendto(value.encode(), (udp_ip, udp_port))



if __name__ == '__main__':
    main()