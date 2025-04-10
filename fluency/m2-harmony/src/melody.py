import random as r

import mido
import numpy as np
from mido import MidiFile, MidiTrack, Message

import subprocess
from pydub import AudioSegment

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

    def generate_with_constant_rests(self, octave: int = 1, sig: int = 4, sub: int = 2, length: int = 2, resolve: bool = True):
        tonic = self.root + 12 * octave
        dur = 1 / sub
        melody = [tonic, -1]
        prev = tonic

        last_note_idx = 0
        # First half
        for _ in range(int(length / 2 * sig * sub - 1)):
            spaces = len(melody) - last_note_idx + 1

            curr = r.choice([n for n in self.notes if abs(n - prev) <= 4 and n != prev])
            last_note_idx = len(melody)
            prev = curr
            melody.append(curr)
            melody.append(-1)  # Ensure a rest after each note

        # Second half
        for _ in range(int(length / 2 * sig * sub)):
            spaces = len(melody) - last_note_idx + 1

            steps = [1, 0, -1, -2, -2, -3] if prev - tonic > 0 else [-1, 0, 1, 2, 2, 3]
            curr = prev + r.choice(steps)
            while curr not in self.notes or curr == tonic:
                curr = prev + r.choice(steps)
            last_note_idx = len(melody)
            prev = curr
            melody.append(curr)
            melody.append(-1)  # Ensure a rest after each note

        if resolve:
            melody.append(tonic)
            melody.append(-1)  # Final rest

        return '|'.join(f'{md.int_to_note(val)},{dur}' for val in melody)

    def to_midi(self, melody: str, filename: str = "melody.mid", tempo: int = 120):
        midi_file = MidiFile()
        track = MidiTrack()
        midi_file.tracks.append(track)

        # Convert BPM to MIDI ticks
        ticks_per_beat = midi_file.ticks_per_beat
        microseconds_per_beat = int(60_000_000 / tempo)
        track.append(mido.MetaMessage('set_tempo', tempo=microseconds_per_beat))

        # Parse melody
        notes = melody.split('|')
        for note in notes:
            note_val, dur = note.split(',')
            duration = int(float(dur) * ticks_per_beat)  # Convert duration to ticks

            if note_val == 'X':  # Rest
                track.append(Message('note_off', note=0, velocity=0, time=duration))
            else:
                midi_note = md.note_to_int(note_val) + 60  # Convert note name to MIDI int
                track.append(Message('note_on', note=midi_note, velocity=64, time=0))
                track.append(Message('note_off', note=midi_note, velocity=64, time=duration))

        # Save file
        midi_file.save(filename)
        print(f"MIDI file saved as {filename}")


class HarmonyGenerator:
    def __init__(self, key='C'):
        self.key = key
        self.chord_tensions = md.CHORD_TENSIONS
        self.chord_families = md.CHORD_FAMILIES

    def generate_chord_progression(self, target_tensions, delta_target=0.0, rest=0.5, lambda_balance=1.0):
        """
        delta_target: desired delta tension between consecutive chords
        lambda_balance: weight factor to balance tension vs. transition delta
        """
        progression = []
        previous_chord = None
        for idx, target_tension in enumerate(target_tensions):
            possible_chords = self._generate_possible_chords()
            weights = []
            for chord, chord_tension in possible_chords:
                # Compute tension error: difference between candidate chord's tension and target tension.
                tension_error = abs(chord_tension - target_tension)

                # If there is a previous chord, compute the measured delta tension.
                if previous_chord is not None:
                    measured_delta = self._calculate_transition_delta(previous_chord, chord)
                    target = target_tensions[idx] - target_tensions[idx - 1] if idx > 0 else 0
                    delta_error = abs(measured_delta - target)
                else:
                    delta_error = 0.0

                # Composite error: balance the two metrics
                composite_error = tension_error + lambda_balance * delta_error

                # Convert error into a weight (using a smoothing constant epsilon = 0.01)
                weight = 1 / ((composite_error + 0.01) ** 3)
                weights.append(weight)

            # Choose chord based on composite weight
            selected_chord = r.choices(possible_chords, weights=weights, k=1)[0][0]
            progression.append(selected_chord)
            print('enhanced', selected_chord)
            if previous_chord:
                print(self._calculate_transition_delta(previous_chord, selected_chord))
            previous_chord = selected_chord
        return self.to_str(progression, 2, rest)

    def _generate_possible_chords(self):
        return list(self.chord_tensions.items())

    def _calculate_transition_delta(self, chord_a, chord_b):
        """
        Computes the delta tension between chord_a and chord_b as defined:
        D(a, b) = |T(a) - T(b)| / sum_{i in chord_a, j in chord_b} (i - j)^2

        chord_a and chord_b are expected to be strings like 'C_M' (where the note and chord type are separated by '_').
        The chord structure (intervals) is fetched from md.HARMONIES_SHORT.
        """
        tension_a = self.get_chord_tension(chord_a)
        tension_b = self.get_chord_tension(chord_b)
        tension_diff = abs(tension_a - tension_b)
        sign = np.sign(tension_a - tension_b)

        # Extract base notes and chord type identifiers.
        note_a, chord_type_a = chord_a.split('_')
        note_b, chord_type_b = chord_b.split('_')
        base_a = md.note_to_int(note_a + '0')
        base_b = md.note_to_int(note_b + '0')
        intervals_a = md.HARMONIES_SHORT[chord_type_a]
        intervals_b = md.HARMONIES_SHORT[chord_type_b]

        # Compute the chord pitches.
        chord_a_notes = [base_a + interval for interval in intervals_a]
        chord_b_notes = [base_b + interval for interval in intervals_b]

        # Compute the denominator as the sum of squared differences between all pairs of notes.
        # (Make sure to handle the case where the denominator might be zero.)
        diff = np.sqrt(sum((i - j) ** 2 for i, j in zip(chord_a_notes, chord_b_notes)))

        if diff == 0:
            return 0.0
        # print('DELTA:')
        # print(tension_diff, diff)

        raw_tension = tension_diff * diff
        res = sign * raw_tension / (1 + raw_tension)
        # print(res)
        return res

    def get_chord_tension(self, chord):
        return self.chord_tensions.get(chord, 0.0)

    def get_chord_family(self, chord):
        for family, chords in self.chord_families.items():
            if chord in chords:
                return family
        return 'unknown'

    def enhance_tension(self, original_progression, delta_t, lambda_balance=1.0):
        enhanced_progression = []
        previous_chord = None
        for chord in original_progression:
            current_tension = self.get_chord_tension(chord)
            new_tension = current_tension + delta_t
            possible_chords = self._generate_possible_chords()
            weights = []
            for candidate, candidate_tension in possible_chords:
                tension_error = abs(candidate_tension - new_tension)
                if previous_chord is not None:
                    measured_delta = self._calculate_transition_delta(previous_chord, candidate)
                    delta_error = abs(measured_delta - delta_t)
                else:
                    delta_error = 0.0
                composite_error = tension_error + lambda_balance * delta_error
                weight = 1 / ((composite_error + 0.01) ** 3)
                weights.append(weight)
            enhanced_chord = r.choices(possible_chords, weights=weights, k=1)[0][0]
            enhanced_progression.append(enhanced_chord)
            previous_chord = enhanced_chord
        return enhanced_progression

    def get_chords_by_tension(self, tension_ranges):
        """Generate chords based on the predefined tension ranges."""

        target_tensions = [r.uniform(tension_ranges[i], tension_ranges[i + 1]) for i in
                           range(len(tension_ranges) - 1)]
        return self.generate_chord_progression(target_tensions, rest=0)

    @staticmethod
    def to_str(chords, length, rest):
        res = []
        for idx, expr in enumerate(chords):
            note_str, chord_str = expr.split('_')
            note_str += '0' if idx < len(chords) - 1 else '0'
            note_int = md.note_to_int(note_str)
            chord_int = [note_int + i for i in md.HARMONIES_SHORT[chord_str]]
            res.append(f'{[md.int_to_note(i) for i in chord_int]}-{length - rest}')
            res.append(f'X-{rest}')

        return '|'.join(res)





def midi_to_mp3(midi_file: str, output_mp3: str, soundfont: str = "default.sf2"):
    """
    Converts a MIDI file to MP3 using FluidSynth.

    :param midi_file: Path to the input MIDI file.
    :param output_mp3: Path to save the output MP3 file.
    :param soundfont: Path to a SoundFont (.sf2) file.
    """
    wav_file = output_mp3.replace(".mp3", ".wav")

    # Convert MIDI to WAV using FluidSynth
    subprocess.run(["fluidsynth", "-ni", soundfont, midi_file, "-F", wav_file, "-r", "44100"], check=True)

    # Convert WAV to MP3
    audio = AudioSegment.from_wav(wav_file)
    audio.export(output_mp3, format="mp3", bitrate="192k")
    print(f"MP3 file saved as: {output_mp3}")

if __name__ == '__main__':
    hg = HarmonyGenerator()
    mg = MelodyGenerator(scale='aeolian')
    # temp = mg.generate_with_constant_rests(1, length=1)
    # mg.to_midi(temp)
    # midi_to_mp3('melody.mid', 'melody.mp3', '../resources/piano.sf2')
    temp = hg.generate_chord_progression([0.1, 0.2, 0.6, 0.9])
    print(type(temp))
    print(hg.to_str(temp, 2, 0.5))