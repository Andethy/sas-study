import time

from pythonosc import udp_client

# import config
from common import utils

class Robot(utils.robotsUtils):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_osc(self):
        self.client = udp_client.SimpleUDPClient(self.IPtoSEND, 25251)

        amp = [1, 0, 0, 0, 0, 0, 0, 0,
               1, 0, 0, 0, 0, 0, 0, 0,
               1, 0, 0, 0, 0, 0, 0, 0,
               1, 0, 0, 0, 0, 0, 0, 0,
               1, 0, 0, 0, 1, 0, 0, 0,
               1, 0, 0, 0, 1, 0, 0, 0,
               1, 0, 1, 0, 1, 0, 1, 0,
               1, 0, 1, 0, 1, 0, 1, 0]

        pitch =[*([0] * 8), *([0.5] * 8), *([0] * 8), *([0.5] * 8), *([0.5*5/12] * 8), *([0.5] * 8), *([0.5*5/12] * 8), *([0.5*13/12] * 8)]

        for a, p in zip(amp, pitch):
            print(a, p)
            self.client.send_message("/rhythm",float(a))
            self.client.send_message("/melody", float(p))
            time.sleep(0.1)
