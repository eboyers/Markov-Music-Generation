import random
from collections import defaultdict, Counter

from theory import MusicTheory

class MarkovModel:    
    def __init__(self, order=5):
        """
        Initialize Markov Chain model; order of the Markov chain is how many 
        previous notes to consider.
        """
        self.order = order
        self.transitions = defaultdict(Counter)
        self.composers_transitions = {}
        self.rhythm_transitions = defaultdict(Counter)
        self.phrase_beginnings = []  # store phrase starting points
        self.phrase_endings = []     # store phrase ending points
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
                    # this is a good place for a phrase ending
                    self.phrase_endings.append(tuple(sequence[i - self.order: i]))
                    
                # this is a good place for beginning a phrase
                self.phrase_beginnings.append(tuple(sequence[i: i + self.order]))
                
            # train the Markov chain
            for i in range(len(sequence) - self.order):
                # create a tuple of the current state (previous notes)
                state = tuple(sequence[i:i + self.order])
                next_note = sequence[i + self.order] # get the next note
                self.transitions[state][next_note] += 1 # update transition count
                
                # extract rhythmic patterns
                if i > 0:
                    intervals = []
                    for j in range(1, len(state)):
                        intervals.append(state[j] - state[j-1])
                    
                    # record the interval patterns
                    interval_state = tuple(intervals)
                    next_interval = next_note - state[-1]
                    self.rhythm_transitions[interval_state][next_interval] += 1
    
    def train_by_composer(self, all_sequences):
        """
        Train separate Markov models for each composer.
        """
        for composer, sequences in all_sequences.items():
            composer_model = MarkovModel(self.order)
            composer_model.train(list(sequences.values()))
            self.composers_transitions[composer] = composer_model.transitions
    
    def generate_melody(self, key, scale_type, melody_length=32, phrase_length=4):
        """
        Generate melody using Markov chain transitions, respecting musical structure and theory.
        """
        scale_notes = self.theory.get_scale_notes(key, scale_type)
        chord_progression = self.theory.generate_chord_progression(scale_type)
        notes_per_chord = phrase_length // len(chord_progression)
        
        # use chord tones from the first chord in the progression for initialization
        start_notes = self.theory.get_chord_notes(key, chord_progression[0], scale_type)
        while len(start_notes) < self.order: # pad with more notes from scale if necessary
            start_notes.append(random.choice(scale_notes))
        
        start_notes = start_notes[:self.order]  # trim to match order
        melody = start_notes.copy()  # init melody with start notes
        
        durations = [480] * self.order # durations of each note in melody, all quarter for simplicity
        
        # make melody phrase by phrase
        current_phrase = 0
        phrase_start_idx = 0
        
        while len(melody) < melody_length + self.order:
            # determine which chord we're on within the current phrase
            phrase_position = (len(melody) - self.order - phrase_start_idx) 
            chord_index = (phrase_position // notes_per_chord) % len(chord_progression)
            current_chord = chord_progression[chord_index]
            
            # get chord tones for the current chord
            chord_notes = self.theory.get_chord_notes(key, current_chord, scale_type)
            
            # get the current state
            state = tuple(melody[-self.order:])
            
            # check if we should start a new phrase
            if phrase_position >= phrase_length:
                current_phrase += 1
                phrase_start_idx = len(melody) - self.order
                
                # use a good melodic transition between phrases if available
                if self.phrase_beginnings and random.random() < 0.7:  # 70% chance to use phrase transitions
                    closest_beginning = min(self.phrase_beginnings, 
                                        key=lambda x: sum(abs(a-b) for a, b in zip(x, state)))
                    
                    # If it's a reasonable transition, use it
                    if sum(abs(a-b) for a, b in zip(closest_beginning, state)) < 20:
                        for note in closest_beginning:
                            # Ensure the note is in our key
                            adjusted_note = self.theory.get_nearest_scale_note(note, key, scale_type)
                            melody.append(adjusted_note)
                            durations.append(480)  # uarquarteter note for each transition note
            
            next_note = None # Markov logic
            
            # use the Markov model transition probabilities
            if state in self.transitions and self.transitions[state]:
                # convert Counter to probability distribution
                choices, weights = zip(*self.transitions[state].items())
                total_weight = sum(weights)
                if total_weight > 0: # valid transitions
                    # normalize weights
                    probs = [w / total_weight for w in weights]
                    
                    # keep trying until note in scale
                    while True:
                        candidate_note = random.choices(choices, weights=probs, k=1)[0]
                        if self.theory.is_in_key(candidate_note, key, scale_type):
                            next_note = candidate_note
                            break
            
            # add next note to the melody
            melody.append(next_note)
            durations.append(480)  # quarter note duration
        
        # trim to required length
        return melody[self.order:self.order+melody_length], durations[:melody_length]