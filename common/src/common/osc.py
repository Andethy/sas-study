from typing import Dict, Any

from pythonosc import udp_client


class OSCManager:

    ports: dict[int, udp_client.SimpleUDPClient]

    def __init__(self, address, *ports, base=-1, k=0, **kwargs):
        self.address = address
        self.ports = {}

        # Case A: Set ports specified in args
        for port in ports:
            self.ports[port] = udp_client.SimpleUDPClient(address, port)

        # Case B: Ascending from base e.g. 1000, 1001, 1002, ... x k
        for i in range(k):
            self.ports[base + i] = udp_client.SimpleUDPClient(address, base + i)

        # Case C: custom port setup
        for port, client in kwargs.items():
            # noinspection PyTypeChecker
            self.ports[port] = client

    def __getitem__(self, item):
        return self.ports[item]

    def __setitem__(self, key, value):
        self.ports[key].send_message(key, float(value))

