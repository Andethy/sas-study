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
    "jazz-minor": (0, 2, 3, 5, 7, 9, 11)
}

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
    "minor-9th": (0, 3, 7, 10, 14)
}

HARMONIES_SHORT = {
    "M": (0, 4, 7, 12),
    "m": (0, 3, 7, 12),
    "dim": (0, 3, 6, 0),
    "aug": (0, 4, 8, 0),
    "sus2": (0, 2, 7, 12),
    "sus4": (0, 5, 7, 12),
    "M6": (0, 4, 7, 9, 12),
    "M7": (0, 4, 7, 11),
    "m7": (0, 3, 7, 10),
    "dom7": (0, 4, 7, 10),
    "dom7sus4": (0, 5, 7, 10),
    "dom7b5": (0, 4, 8, 10),
    "dom7aug5": (0, 5, 8, 10),
    "m7b5": (0, 3, 6, 10),
    "dim7": (0, 3, 6, 9),
    "M9": (0, 4, 7, 11, 14),
    "m9": (0, 3, 7, 10, 14)
}

HARMONY_KEYS = HARMONIES.keys()

UNUSED = {
    "perfect-5th": (0, 7),
}

# New additions based on the research paper
# CHORD_TENSIONS = {
#     # Tonic
#     'C_M': 0.0, 'C_M7': 0.07, 'C_M6': 0.21, 'A_m': 0.50,
#     'A_m7': 0.57, 'E_m': 0.57, 'E_m7': 0.64,
#     # Subdominant
#     'F_M': 0.36, 'F_M7': 0.43, 'F_M6': 0.57, 'D_m': 0.57,
#     'D_m7': 0.64,
#     # Dominant
#     'G_M': 0.36, 'G_dom7': 0.43, 'G_dom7sus4': 0.57, 'B_dim': 0.79,
#     # 'G_dom7b5': 0.79, 'G_dom7aug5': 0.79,
#     'B_dom7b5': 0.86, 'B_dim7': 0.86,
# }

CHORD_TENSIONS = {'C_M': 0.0456,
 'C_M7': 0.20720000000000002,
 'C_M6': 0.10760000000000002,
 'A_m': 0.2632,
 'A_m7': 0.3776,
 'E_m': 0.1744,
 'E_m7': 0.2888,
 'F_M': 0.43379999999999996,
 'F_M7': 0.5954,
 'F_M6': 0.4958,
 'D_m': 0.3832,
 'D_m7': 0.4976,
 'G_M': 0.6264,
 'G_dom7': 0.76,
 'G_dom7sus4': 0.7568,
 'B_dim': 0.8376000000000001,
 'B_dom7b5': 0.7696000000000001,
 'B_dim7': 0.8592000000000001}

CHORD_TENSIONS_FULL = {
    # Tonic
    'C_M': 0.0, 'C_M7': 0.07, 'C_M6': 0.21, 'A_m': 0.50,
    'A_m7': 0.57, 'E_m': 0.57, 'E_m7': 0.64, 'F#_m7b5': 0.86,
    # Subdominant
    'F_M': 0.36, 'F_M7': 0.43, 'F_M6': 0.57, 'D_m': 0.57,
    'D_m7': 0.64, 'A#_M7': 0.71,
    # Dominant
    'G_M': 0.36, 'G_dom7': 0.43, 'G_dom7sus4': 0.57, 'B_dim': 0.79,
    'G_dom7b5': 0.79, 'G_dom7aug5': 0.79, 'B_dom7b5': 0.86, 'B_dim7': 0.86,
    'C#_dom7': 1.00
}

CHORD_FAMILIES = {
    'tonic': ['I', 'CM7', 'C6', 'Am', 'Am7', 'Em', 'Em7', 'F#m7-5'],
    'subdominant': ['F', 'FM7', 'F6', 'Dm', 'Dm7', 'BbM7'],
    'dominant': ['G', 'G7', 'G7sus4', 'Bm-5', 'G7-5', 'G7+5', 'Bm7-5', 'Bdim7', 'Db7']
}

def get_key_notes(root, scale):
    key_int = get_key_ints(root, scale)
    key_str = []
    for note in key_int:
        key_str.append(TONICS_INT[note % 12])
    return list(TONICS_INT[note % 12] for note in key_int)

def get_key_ints(root, scale) -> list[int]:
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

# New function to get chord tension
def get_chord_tension(chord):
    return CHORD_TENSIONS.get(chord, 0.0)

# New function to get chord family
def get_chord_family(chord):
    for family, chords in CHORD_FAMILIES.items():
        if chord in chords:
            return family
    return 'unknown'
