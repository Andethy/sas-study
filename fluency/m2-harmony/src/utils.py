import ast
import collections
import time as t

import numpy as np
from pythonosc import udp_client

# import config
from common import utils as ut, midi as md
from common.osc import OSCManager
from constants import SYNTH_RANGE, ADSR_PORT, NOTE_PORT, ON_PORT
from melody import HarmonyGenerator


class EnvelopeGenerator:

    def __init__(self, attack=0.001, release=0.2):
        self.attack = attack
        self.release = release

    def __call__(self, n, x, *args, **kwargs):
        return float(min(x / self.attack, 1) if n != 'X' else max(1 - x / self.release, 0))



class StaticRobot(ut.robotsUtils):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, 25251, 25252, **kwargs)
        self.eg = EnvelopeGenerator(attack=0.01, release=0.5)

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
            if temp > 15:
                temp %= 12
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

        last_on = 0
        while elapsed < time:
            if elapsed > melody[idx][-1]:
                idx += 1

            note, on, off = melody[idx]



            if note != 'X':
                last_on = off
                self.client.send_message(ADSR_PORT, self.eg(note, elapsed - on))
                self.client.send_message(NOTE_PORT, ntm(note))
            else:
                self.client.send_message(ADSR_PORT, self.eg(note, elapsed - last_on))


            print(ntm(note), self.eg(note, elapsed - on))

            t.sleep(0.004)
            elapsed = t.time() - start

        self.client.send_message(ADSR_PORT, 0.0)

    def harmony(self, time):
        melodies = {
            25251: collections.deque(extract_notes('C1,1|D1,0.5|X,0.5|C1,1|D1,0.5|X,0.5|D#1,1|F1,0.5|X,0.5|G1,1|D#1,0.5|X,0.5|C1,1|X,1', time)),
            25252: collections.deque(extract_notes('C0,2|D#0,2|G0,2|G#0,2|G0,1|X,1', time))
        }

        # self.client = udp_client.SimpleUDPClient(self.IPtoSEND, 25251)

        start = t.time()
        elapsed = 0
        last_on = 0

        idx = 0

        while elapsed < time:
            for port in melodies:
                if len(melodies[port]) > 1 and elapsed > melodies[port][0][-1]:
                    melodies[port].popleft()

                note, on, off = melodies[port][0]
                if note != 'X':
                    self.clients(port, ADSR_PORT, 1)
                    self.clients(port, NOTE_PORT, ntm(note))
                else:
                    self.clients(port, ADSR_PORT, 0)

            t.sleep(0.004)
            elapsed = t.time() - start

    def from_curve(self, curve = (0.1, 0.3, 0.7, 0.9), time=5):
        hg = HarmonyGenerator()

        harmony = extract_chords(hg.generate_chord_progression(curve, rest=0., lambda_balance=0.1), time)

        osc = OSCManager(self.IPtoSEND, base=25251, k=4)

        start = t.time()
        elapsed = 0
        last_on = 0

        idx = 0

        while elapsed < time:
            if elapsed > harmony[idx][-1]:
                idx += 1

            notes, on, off = harmony[idx]

            for i, port in enumerate(osc):
                if notes != 'X':
                    last_on = off
                    osc[port, ADSR_PORT] = self.eg(notes[i], elapsed - on)
                    temp = ntm(notes[i])
                    if temp > 15:
                        temp -= 12
                    osc[port, NOTE_PORT] = temp
                else:
                    osc[port, ADSR_PORT] = self.eg(notes, elapsed - last_on)
            print(notes, on, off)
            t.sleep(0.004)
            elapsed = t.time() - start
        t.sleep(1.0)
        for port in osc:
            osc[port, ADSR_PORT] = 0

class DynamicRobot(ut.robotsUtils):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eg = EnvelopeGenerator(attack=0.01, release=0.5)
        self.osc = OSCManager(self.IPtoSEND, base=25251, k=4)
        for port in self.osc:
            self.osc[port, ON_PORT] = 1
            self.osc[port, ADSR_PORT] = 0
        self.chords = None


    def activate(self, tensions):

        hg = HarmonyGenerator()
        chords = extract_chords(hg.get_chords_by_tension(tensions), 10)
        self.chords = [chord[0] for chord in chords[::2]]

        print(self.chords)

        for port in self.osc:
            self.osc[port, ADSR_PORT] = 1

    def deactivate(self):
        for port in self.osc:
            self.osc[port, ADSR_PORT] = 0

    def set_tension(self, index):
        for port, note in zip(self.osc, self.chords[index][:4]):
            print(f'Setting port {port} to {note}')
            self.osc[port, NOTE_PORT] = ntm(note)

    def set_chord_by_tension(self, tension, prev_chord, hg: HarmonyGenerator, lambda_balance=1):
        """
        Sets robot OSC outputs based on a new chord selected from the target tension.
        """
        new_chord = hg.select_chord_by_tension(tension, prev_chord, lambda_balance)
        print(new_chord)
        notes = hg.chord_to_list(new_chord)
        print('??', notes)
        notes = [ntm(note) for note in notes]
        print('???', notes)

        for port, note in zip(self.osc, notes):
            print(f"Setting port {port} to {note} (from chord {new_chord})")
            self.osc[port, NOTE_PORT] = note

        return new_chord


def extract_notes(file, time) -> list:
    try:
        with open(file) as f:
            res = [line.split() for line in f.readlines()]
    except FileNotFoundError:
        res = [line.split(',') for line in file.split('|')]

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

def extract_chords(expr, time) -> list:
    curr = str(expr.split('|')[0].split('-')[0])
    print('hmm', curr)
    print(ast.literal_eval(curr))
    res = [[ast.literal_eval(str(line.split('-')[0].replace(' ', ''))) if line.split('-')[0] != 'X' else 'X', line.split('-')[1]]  for line in expr.split('|')]


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
    if tone not in md.TONICS_STR or not octave.isnumeric() or int(octave) < 0 or int(octave) >= SYNTH_RANGE / 12:
        raise ValueError(f"Invalid tone {tone} or octave {octave}")

    temp = md.TONICS_STR[tone]
    octave = int(octave)
    if temp > 5 and octave > 0:
        octave -= 1
        print('reducing')
    return (temp + 12 * (octave)) / SYNTH_RANGE