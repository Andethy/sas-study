import time

from pythonosc import udp_client

# import config
from common import utils

class Robot(utils.robotsUtils):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)