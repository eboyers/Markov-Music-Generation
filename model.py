import random

from collections import defaultdict, Counter
from theory import MusicTheory

class MarkovModel:    
    def __init__(self, order):
        """Initialize Markov Chain model; order of the Markov chain is how many previous notes to consider."""
        self.order = order
        self.transitions = defaultdict(Counter)
        self.composers_transitions = {}
        self.phrase_beginnings = self.phrase_endings = []  # store phrase starting/ending points
        self.theory = MusicTheory()  # ingrain theory into model
        
    def train(self, note_sequences, phrase_length):
        """
        Train the Markov model on note sequences.
        - note_sequences is list of note sequences
        - phrase_length is typical phrase length
        """
        for sequence in note_sequences:
            if len(sequence) <= self.order:
                continue
            
            # extract phrase beginnings and endings
            for i in range(0, len(sequence) - phrase_length, phrase_length):
                if i > 0:
                    self.phrase_endings.append(tuple(sequence[i - self.order: i])) # this is a good place for a phrase ending
                self.phrase_beginnings.append(tuple(sequence[i: i + self.order])) # this is a good place for beginning a phrase
                
            # train the Markov chain
            for i in range(len(sequence) - self.order):
                # create a tuple of the current state (previous notes)
                state = tuple(sequence[i: i + self.order])
                next_note = sequence[i + self.order] # get the next note
                self.transitions[state][next_note] += 1 # update transition count
    
    def generate_melody(self, key, scale_type, melody_length=32, phrase_length=4):
        """Generate melody using Markov chain transitions, respecting musical structure and theory."""
        scale_notes = self.theory.get_scale_notes(key, scale_type)
        chord_progression = self.theory.generate_chord_progression(scale_type)
        notes_per_chord = phrase_length // len(chord_progression)
        
        # use chord tones from the first chord in the progression for initialization
        start_notes = self.theory.get_chord_notes(key, chord_progression[0], scale_type)
        while len(start_notes) < self.order: # pad with more notes from scale if necessary
            start_notes.append(random.choice(scale_notes))
        
        start_notes = start_notes[:self.order] # trim to match order
        melody = start_notes.copy() # init melody with start notes
        
        durations = [480] * self.order # durations of each note in melody, all quarter for simplicity
        
        current_phrase = phrase_start_idx = 0 # create melody phrase by phrase
        
        while len(melody) < melody_length + self.order:
            # determine which chord we're on within the current phrase
            phrase_position = (len(melody) - self.order - phrase_start_idx) 
            chord_index = (phrase_position // notes_per_chord) % len(chord_progression)
            current_chord = chord_progression[chord_index]
            
            chord_notes = self.theory.get_chord_notes(key, current_chord, scale_type) # get chord tones for the current chord
            state = tuple(melody[-self.order:]) # get the current state
            
            # check if we should start a new phrase
            if phrase_position >= phrase_length:
                current_phrase += 1
                phrase_start_idx = len(melody) - self.order
            
            next_note = None # Markov logic
            
            # use the Markov model transition probabilities
            if state in self.transitions and self.transitions[state]:
                # convert Counter to probability distribution
                choices, weights = zip(*self.transitions[state].items())
                total_weight = sum(weights)
                if total_weight > 0: # valid transitions
                    probs = [w / total_weight for w in weights] # normalize weights
                    attempts = 0
                    max_attempts = 10
                    while attempts < max_attempts: # keep trying until note in scale
                        candidate_note = random.choices(choices, weights=probs, k=1)[0]
                        if self.theory.is_in_key(candidate_note, key, scale_type):
                            next_note = candidate_note
                            break
                        attempts += 1
            
            if next_note is None: # next note can't be none
                # end of phrase or on chord change - prefer using chord tones
                if phrase_position % notes_per_chord == 0 or phrase_position == phrase_length - 1:
                    next_note = random.choice(chord_notes)
                else:
                    last_note = melody[-1] # use a scale note with preference for smoother voice leading
                    close_notes = [n for n in scale_notes if 0 < abs(n - last_note) <= 5] # find scale notes that are close to the last note
                    if close_notes:
                        next_note = random.choice(close_notes)
                    else:
                        next_note = random.choice(scale_notes)
            
            melody.append(next_note) # add the next note to the melody
            durations.append(480) # quarter note duration for simplicity

        return melody[self.order: self.order + melody_length], durations[: melody_length] # trim to specific length
