import random

SCALES = { # scales as semitones up from root
        'major': [0, 2, 4, 5, 7, 9, 11],
        'natural_minor': [0, 2, 3, 5, 7, 8, 10],
        'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
        'melodic_minor': [0, 2, 3, 5, 7, 9, 11],  # ascending melodic minor
    }
    
PROGRESSIONS = { # common chord progressions (as scale degrees)
    'major': [
        [1, 4, 5, 1],  # I-IV-V-I
        [1, 6, 4, 5],  # I-vi-IV-V
        [1, 4, 6, 5],  # I-IV-vi-V
        [6, 2, 5, 1],  # vi-ii-V-I
        [1, 5, 6, 4],  # I-V-vi-IV
    ],
    'minor': [
        [1, 4, 5, 1],  # i-iv-v-i
        [1, 6, 3, 7],  # i-VI-III-VII
        [1, 7, 6, 5],  # i-VII-VI-v
        [1, 4, 7, 3],  # i-iv-VII-III
        [6, 7, 3, 5],  # VI-VII-III-v
    ]
}

class MusicTheory:
    """
    Music theory basics for more informed melodic generation.
    """
    @staticmethod
    def get_scale_notes(key, scale_type):
        """
        Get notes in a scale given key and scale type.
        """
        base_note = key % 12  # normalize to octave
        intervals = SCALES.get(scale_type, SCALES['major']) # default major
        
        # generate scale notes across MIDI range (36 - 96 or C2 - C7)
        scale_notes = []
        for octave in range(3, 8):
            for interval in intervals:
                note = (octave * 12) + base_note + interval
                if 36 <= note <= 96:
                    scale_notes.append(note)
        
        return sorted(scale_notes)
    
    @staticmethod
    def get_chord_notes(key, scale_degree, scale_type):
        """
        Get the notes in a chord based on scale degree. 
        """
        scale = MusicTheory.get_scale_notes(key, scale_type)
        
        # adjust for 1-based indexing of scale degrees
        degree_idx = scale_degree - 1
        
        # get chord root note from the scale
        root_idx = degree_idx

        # triad intervals are 1, 3, 5
        third_idx = (root_idx + 2) % 7
        fifth_idx = (root_idx + 4) % 7
        
        # extract actual notes from the full scale
        triad_root = [n for n in scale if n % 12 == (key + SCALES[scale_type][root_idx]) % 12]
        triad_third = [n for n in scale if n % 12 == (key + SCALES[scale_type][third_idx]) % 12]
        triad_fifth = [n for n in scale if n % 12 == (key + SCALES[scale_type][fifth_idx]) % 12]
        
        mid_range = 60 # take the notes closest to the middle octave; keep melody in middle range for simplicity
        root = min(triad_root, key=lambda x: abs(x - mid_range))
        third = min(triad_third, key=lambda x: abs(x - mid_range))
        fifth = min(triad_fifth, key=lambda x: abs(x - mid_range))
        
        return [root, third, fifth] # complete triad
    
    @staticmethod
    def is_in_key(note, key, scale_type):
        """
        Check if note belongs to a specific key.
        """
        scale_notes = MusicTheory.get_scale_notes(key, scale_type)
        return note % 12 in [sn % 12 for sn in scale_notes]
    
    @staticmethod
    def get_nearest_scale_note(note, key, scale_type):
        """
        Get nearest note that belongs to the scale.
        """
        if MusicTheory.is_in_key(note, key, scale_type):
            return note
            
        scale_notes = MusicTheory.get_scale_notes(key, scale_type)
        return min(scale_notes, key=lambda x: abs(x - note))
    
    @staticmethod
    def generate_chord_progression(scale_type):
        """
        Generate random chord progression for a given key.
        """
        if scale_type.endswith('minor'): progressions = PROGRESSIONS['minor']
        else: progressions = PROGRESSIONS['major']
        return random.choice(progressions)