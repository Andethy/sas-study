from typing import Dict, Any

from pythonosc import udp_client


class OSCManager:

    ports: dict[int, udp_client.SimpleUDPClient]

    def __init__(self, address, *ports, base=-1, k=0, **kwargs):
        self.address = address
        self.ports = {}

        # Case A: Set ports specified in args
        for port in ports:
            print(f'Setup client on port {port} (de)')
            self.ports[port] = udp_client.SimpleUDPClient(address, port)

        # Case B: Ascending from base e.g. 1000, 1001, 1002, ... x k
        for i in range(k):
            print(f'Setup client on port {base + i} (bi)')
            self.ports[base + i] = udp_client.SimpleUDPClient(address, base + i)

        # Case C: custom port setup
        for port, client in kwargs.items():
            print(f'Setup client on port {port} (dc)')
            # noinspection PyTypeChecker
            self.ports[port] = client

    def __call__(self, port, addr, val, *args, **kwargs):
        print(f'Sending msg on port {port} to {addr} set to {val}')
        self[port].send_message(addr, float(val))

    def __getitem__(self, item):
        return self.ports[item]

    def __setitem__(self, key, value):
        self.ports[key].send_message(key, float(value))

    def reconnect(self):
        for port in self.ports:
            self.ports[port] = udp_client.SimpleUDPClient(self.address, port)
