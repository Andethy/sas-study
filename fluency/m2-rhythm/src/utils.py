import collections
import time as t

import numpy as np
from pythonosc import udp_client

# import config
from common import utils as ut, midi as md
from constants import SYNTH_RANGE, ADSR_PORT, NOTE_PORT


class EnvelopeGenerator:

    def __init__(self, attack=0.05, release=0.25):
        self.attack = attack
        self.release = release

    def __call__(self, n, x, *args, **kwargs):
        return float(min(x / self.attack, 1) if n != 'X' else max(1 - x / self.release, 0))



class Robot(ut.robotsUtils):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eg = EnvelopeGenerator()

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
            self.client.send_message(ADSR_PORT,float(a))
            self.client.send_message(NOTE_PORT, float(p))
            t.sleep(0.1)

    def test_scale(self):
        """
        NOTE: This is hard coded right now.
        TODO: Make a generalized scale version.
        """
        TMP = 64

        a = [0] * TMP
        p = []

        note_seq = collections.deque(['A#0', 'D#0', 'C#0', 'G0', 'E0', 'B0', 'F#0', 'C#1'])
        # octave_seq = collections.deque(([0] * 7) + [0])

        # I think it will make a blip-type sound ascending c scale
        for i in range(0, TMP, 8):
            a[i] = 1
            a[i + 1] = 1
            temp = ntm(note_seq.popleft())
            p.extend([temp] * 8)

        print(np.array(p))

        self.test_osc(a, p)

    def from_file(self, file, time = 5):
        melody = extract_notes(file, time)

        # Hmm refactor this later TODO this needs to be done
        self.client = udp_client.SimpleUDPClient(self.IPtoSEND, 25251)

        print(melody)

        start = t.time()
        elapsed = 0

        idx = 0
        while elapsed < time:
            if elapsed > melody[idx][-1]:
                idx += 1

            note, on, off = melody[idx]

            self.client.send_message(ADSR_PORT, self.eg(note, elapsed - on))

            if note != 'X':
                self.client.send_message(NOTE_PORT, ntm(note))

            print(ntm(note), self.eg(note, elapsed - on))

            t.sleep(0.004)
            elapsed = t.time() - start

        self.client.send_message(ADSR_PORT, 0.0)





def extract_notes(f_name, time) -> list:
    with open(f_name) as f:
        res = [line.split() for line in f.readlines()]

        # 1st pass: prefix sum
        res[0].append(float(res[0][-1]))
        res[0][-2] = 0

        for i in range(1, len(res)):
            temp = float(res[i][-1])
            res[i][-1] = res[i - 1][-1]
            res[i].append(res[i][-1] + temp)

        # 2nd pass: converting notes to time
        for i in range(len(res)):
            res[i][-2] *= time / res[-1][-1]
            res[i][-1] *= time / res[-1][-1]


    return res



def ntm(note: str) -> float:
    """
    ntm: note to midi

    Only accepts notes in the format [ABCDEFG][#][01]
    """
    if note == 'X':
        return -1

    tone, octave = note[0:-1], note[-1]
    if tone not in md.TONICS_STR or not octave.isnumeric() or int(octave) < 0 or int(octave) > 1:
        raise ValueError(f"Invalid tone {tone} or octave {octave}")

    return (md.TONICS_STR[tone] + 12 * int(octave)) / SYNTH_RANGE