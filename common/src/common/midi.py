TONICS_STR = {b: a for a, b in enumerate(('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'))}
TONICS_INT = {a: b for a, b in enumerate(('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'))}

SCALES = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "ionian": (0, 2, 4, 5, 7, 9, 11),
    "aeolian": (0, 2, 3, 5, 7, 8, 10),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "harmonic": (0, 2, 3, 5, 7, 8, 11),
    "dominant-phrygian": (0, 1, 4, 5, 7, 8, 10),
    "jazz-minor": (0, 2, 3, 5, 7, 9, 11)}

HARMONIES = {
    "major-triad": (0, 4, 7),
    "minor-triad": (0, 3, 7),
    "dim-triad": (0, 3, 6),
    "aug-triad": (0, 4, 8),
    "sus2-triad": (0, 2, 7),
    "sus4-triad": (0, 5, 7),
    "major-7th": (0, 4, 7, 11),
    "minor-7th": (0, 3, 7, 10),
    "major-9th": (0, 4, 7, 11, 14),
    "minor-9th": (0, 3, 7, 10, 14)}

HARMONY_KEYS = HARMONIES.keys()

UNUSED = {
    "perfect-5th": (0, 7),
}


def get_key_notes(root, scale):
    key_int = get_key_ints(root, scale)
    key_str = []
    for note in key_int:
        key_str.append(TONICS_INT[note % 12])
    return list(TONICS_INT[note % 12] for note in key_int)

def get_key_ints(root, scale):
    root_int = TONICS_STR[root]
    scale_int = SCALES[scale]
    return list(root_int + sc for sc in scale_int)

def note_to_int(note):
    if note == 'X':
        return -1

    tone, octave = note[0:-1], note[-1]
    if tone not in TONICS_STR or not octave.isnumeric() or int(octave) < 0 or int(octave) > 1:
        raise ValueError(f"Invalid tone {tone} or octave {octave}")

    return TONICS_STR[tone] + 12 * int(octave)

def int_to_note(value):
    return TONICS_INT[value % 12] + str(value // 12) if value > -1 else 'X'