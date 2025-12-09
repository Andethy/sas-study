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

    def select_chord_by_tension(
            self,
            current_tension,
            previous_chord,
            lambda_balance=0.3,
            k=3,
            max_tension_error=0.5,
            temperature=0.1,
    ):
        """
        - max_tension_error: hard tolerance for how far from target tension a chord may be
        - temperature: how random the final choice is (0 ≈ deterministic)
        """
        # Special case: for very low tensions, strongly favor C_M
        if current_tension <= 0.005:  # Only force C_M for extremely low tensions
            print(f"[CHORD SELECTION] Very low tension {current_tension:.3f} - forcing C_M")
            return "C_M"
        possible_chords = self._generate_possible_chords()  # [(chord, tension), ...]
        print(f"[CHORD SELECTION] Total possible chords: {len(possible_chords)}")
        
        # Show best tension matches before filtering
        tension_matches = [(chord, tension, abs(tension - current_tension)) for chord, tension in possible_chords]
        tension_matches.sort(key=lambda x: x[2])  # sort by tension error
        print(f"[CHORD SELECTION] Best tension matches:")
        for chord, tension, error in tension_matches[:5]:
            print(f"  {chord}: tension={tension:.3f}, error={error:.3f}")
        
        scored = []

        # 1) first pass: compute separate errors
        for chord, chord_tension in possible_chords:
            # hard constraints first - only skip if same as previous chord
            if chord == previous_chord:
                continue

            tension_error = abs(chord_tension - current_tension)

            # **Hard** tension window – don't even consider outliers
            if tension_error > max_tension_error:
                continue

            if previous_chord is not None:
                target_delta = current_tension - self.get_chord_tension(previous_chord)
                measured_delta = self._calculate_transition_delta(previous_chord, chord)
                delta_error = abs(measured_delta - target_delta)
            else:
                delta_error = 0.0

            # composite error: tension dominates, delta is a secondary term
            composite_error = tension_error + lambda_balance * delta_error
            scored.append((chord, chord_tension, tension_error, delta_error, composite_error))

        # If we filtered too hard and got nothing, relax and fall back
        if not scored:
            # relax: ignore the hard tension window but still rank
            for chord, chord_tension in possible_chords:
                if chord == previous_chord:
                    continue

                tension_error = abs(chord_tension - current_tension)
                if previous_chord is not None:
                    target_delta = current_tension - self.get_chord_tension(previous_chord)
                    measured_delta = self._calculate_transition_delta(previous_chord, chord)
                    delta_error = abs(measured_delta - target_delta)
                else:
                    delta_error = 0.0

                composite_error = tension_error + lambda_balance * delta_error
                scored.append((chord, chord_tension, tension_error, delta_error, composite_error))

        # 2) sort by composite error – this is your main "rule"
        scored.sort(key=lambda x: x[4])  # composite_error

        # 3) keep only the best k candidates
        top = scored[:k]

        # 4) almost deterministic choice:
        #    convert errors to "scores" and sample with low temperature
        errors = np.array([c[4] for c in top], dtype=float)
        # shift so min error ~ 0
        errors = errors - errors.min()
        # turn into energies and then probabilities
        # smaller error -> larger score
        scores = np.exp(-errors / max(temperature, 1e-6))
        probs = scores / scores.sum()

        chosen = r.choices(top, weights=probs, k=1)[0][0]

        # Debug logging
        print(f"[CHORD SELECTION] Target tension: {current_tension:.3f}, Previous: {previous_chord}")
        print(f"[CHORD SELECTION] Found {len(scored)} valid candidates after filtering")
        print(f"[CHORD SELECTION] Top {len(top)} candidates:")
        for chord, ct, te, de, ce in top:
            print(f"  {chord} (T={ct:.3f}): tension_err={te:.3f}, delta_err={de:.3f}, composite={ce:.3f}")
        print(f"[CHORD SELECTION] => CHOSEN: {chosen}")
        
        # Special case logging for very low tensions
        if current_tension <= 0.02:
            print(f"[CHORD SELECTION] LOW TENSION DETECTED: {current_tension:.3f} - Should favor C_M")
        
        # Debug logging for high tensions
        if current_tension >= 0.6:
            print(f"[CHORD SELECTION] HIGH TENSION DETECTED: {current_tension:.3f}")
            print(f"[CHORD SELECTION] Expected candidates: G_M6(0.688), F_M9(0.657), G_dom7sus4(0.757)")
            if chosen in ['C_M9', 'C_M', 'C_M7']:
                print(f"[CHORD SELECTION] WARNING: Selected low-tension chord {chosen} for high tension {current_tension:.3f}!")
        
        # Fallback mechanism: if chosen chord has very poor tension match, override with best tension match
        chosen_tension = self.get_chord_tension(chosen)
        tension_error = abs(chosen_tension - current_tension)
        if tension_error > 0.3:  # If error is too large, use simple best match
            print(f"[CHORD SELECTION] FALLBACK: Tension error {tension_error:.3f} too large, using best tension match")
            # Find best tension match ignoring delta errors
            candidates_by_tension = [(chord, tension, abs(tension - current_tension)) 
                                   for chord, tension in possible_chords 
                                   if chord != previous_chord]
            if candidates_by_tension:
                best_match = min(candidates_by_tension, key=lambda x: x[2])
                chosen = best_match[0]
                print(f"[CHORD SELECTION] FALLBACK CHOSEN: {chosen} (T={best_match[1]:.3f}, error={best_match[2]:.3f})")

        return chosen

    def generate_chord_progression(self, target_tensions, delta_target=0.0, rest=0.5, lambda_balance=1.0, k=4):
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
                weight = 1 / ((composite_error + 0.001) ** 4)
                weights.append(weight)


            largest = np.argpartition(weights, -k)[-4:]

            pc = []
            w = []
            for i in largest:
                pc.append(possible_chords[i])
                w.append(weights[i])

            print('L*', largest)

            # Choose chord based on composite weight
            selected_chord = r.choices(pc, weights=w, k=1)[0][0]
            if target_tension == 0:
                selected_chord = 'C_M'
            progression.append(selected_chord)

            print('TARGET TENSION:', target_tension)
            print(tuple(zip(pc, w)))
            print('ACTUAL TENSION', selected_chord )

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

        note_a, chord_type_a = chord_a.split('_')
        note_b, chord_type_b = chord_b.split('_')
        base_a = md.note_to_int(note_a + '0')
        base_b = md.note_to_int(note_b + '0')
        intervals_a = md.HARMONIES_SHORT[chord_type_a]
        intervals_b = md.HARMONIES_SHORT[chord_type_b]

        chord_a_notes = [base_a + interval for interval in intervals_a]
        chord_b_notes = [base_b + interval for interval in intervals_b]

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

    @staticmethod
    def chord_to_list(chord: str):
        """
        Convert a chord string like 'C_M' into a list of MIDI note integers.
        """
        root_str, chord_type = chord.split('_')
        root_int = md.note_to_int(root_str + '0')  # e.g., C0 → 12, D#0 → 15, etc.
        intervals = md.HARMONIES_SHORT[chord_type]
        return [md.int_to_note(root_int + i) for i in intervals]



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