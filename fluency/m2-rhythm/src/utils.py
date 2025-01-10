import collections
import time

import numpy as np
from pythonosc import udp_client

# import config
from common import utils as ut, midi as md
from constants import SYNTH_RANGE


class Robot(ut.robotsUtils):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_osc(self, amp: list, pitch: list):
        """
        Test rhythmic vital synth patch (REAPER ONLY)


        :param amp: amplitude array over DELTA TIME
        :param pitch: pitch array over DELTA TIME
        """

        # Hmm refactor this later
        self.client = udp_client.SimpleUDPClient(self.IPtoSEND, 25251)

        # amp = [1, 0, 0, 0, 0, 0, 0, 0,
        #        1, 0, 0, 0, 0, 0, 0, 0,
        #        1, 0, 0, 0, 0, 0, 0, 0,
        #        1, 0, 0, 0, 0, 0, 0, 0,
        #        1, 0, 0, 0, 1, 0, 0, 0,
        #        1, 0, 0, 0, 1, 0, 0, 0,
        #        1, 0, 1, 0, 1, 0, 1, 0,
        #        1, 0, 1, 0, 1, 0, 1, 0]
        #
        # pitch =[*([0] * 8), *([0.5] * 8), *([0] * 8), *([0.5] * 8), *([0.5*5/12] * 8), *([0.5] * 8), *([0.5*5/12] * 8), *([0.5*13/12] * 8)]



        for a, p in zip(amp, pitch):
            print(a, p)
            self.client.send_message("/rhythm",float(a))
            self.client.send_message("/melody", float(p))
            time.sleep(0.1)

    def test_scale(self):
        """
        NOTE: This is hard coded right now.
        TODO: Make a generalized scale version.
        """
        TMP = 64

        a = [0] * TMP
        p = []

        note_seq = collections.deque('CDEFGABC')
        pitch_seq = collections.deque(([0] * 7) + [1])

        # I think it will make a blip-type sound ascending c scale
        for i in range(0, TMP, 8):
            a[i] = 1
            temp = ntm(note_seq.popleft() + str(pitch_seq.popleft()))
            p.extend([temp] * 8)

        print(np.array(p))

        self.test_osc()


def ntm(note: str) -> float:
    """
    ntm: note to midi

    Only accepts notes in the format [ABCDEFG][#][01]
    """

    tone, octave = note[:-1], note[-1]
    if tone not in md.TONICS_STR or not octave.isnumeric() or int(octave) < 0 or int(octave) > 1:
        raise ValueError(f"Invalid tone {tone} or octave {octave}")

    return (md.TONICS_STR[tone] + 12 * int(octave)) / SYNTH_RANGE