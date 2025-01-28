import random as r

import common.midi as md
from common.midi import TONICS_STR


class MelodyGenerator:

    def __init__(self, root: str = 'C', scale: str = 'major', octaves: int = 2, length: int = 8):
        self.root = md.TONICS_STR[root]
        self.scale = md.get_key_ints(root, scale)
        self.octaves = octaves
        self.notes = []
        for i in range(octaves):
            self.notes.extend([note + (12 * i) for note in self.scale])
        self.length = length

    def generate(self, octave: int = 1, sig: int = 4, sub: int = 2, length: int = 2):
        tonic = self.root + 12 * octave
        dur = 1 / sub
        melody = [tonic]
        prev = tonic

        last_note_idx = 0
        # First half
        for _ in range(int(length / 2 * sig * sub - 1)):
            spaces = len(melody) - last_note_idx + 1

            if r.random() < 1 / (spaces ** 2):
                melody.append(-1)
                continue

            curr = r.choice([n for n in self.notes if abs(n - prev) <= 4 and n != prev])
            last_note_idx = len(melody)
            prev = curr
            melody.append(curr)

        # Second half
        for _ in range(int(length / 2 * sig * sub)):
            spaces = len(melody) - last_note_idx + 1

            if r.random() < 1 / (spaces ** 2):
                melody.append(-1)
                continue

            steps = [1, 0 -1, -2, -2, -3] if prev - tonic > 0 else [-1, 0, 1, 2, 2, 3]
            curr = prev + r.choice(steps)
            while curr not in self.notes or curr == tonic:
                curr = prev + r.choice(steps)
            last_note_idx = len(melody)
            prev = curr
            melody.append(curr)

        melody.append(tonic)
        melody.append(-1)
        print(melody)

        return '|'.join(f'{md.int_to_note(val)},{dur}' for val in melody)



if __name__ == '__main__':
    mg = MelodyGenerator()
    print(mg.generate(1))