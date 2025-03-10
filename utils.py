import os
import mido
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter, defaultdict
from mido import MidiFile, MidiTrack, Message
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class MIDIProcessor:
    def __init__(self):
        """Class to extract note sequences from MIDI files."""
        self.midi_dir = "classical_midis"
        self.composers = self._get_composers()
        
    def _get_composers(self):
        """Get list of composers from directory structure"""
        return [d for d in os.listdir(self.midi_dir) if os.path.isdir(os.path.join(self.midi_dir, d))]
    
    def extract_notes(self, midi_file_path):
        """Extract a sequence of notes from a MIDI file."""
        notes = []
        mid = MidiFile(midi_file_path)
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    notes.append(msg.note)
        return notes
    
    def get_composer_note_sequences(self, composer):
        """Get note sequences for all MIDI files of a composer."""
        composer_dir = os.path.join(self.midi_dir, composer)
        note_sequences = {}
        
        for file in os.listdir(composer_dir):
            filepath = os.path.join(composer_dir, file)
            notes = self.extract_notes(filepath)
            note_sequences[file] = notes
            
        return note_sequences
    
    def get_all_note_sequences(self):
        """Get note sequences for all composers."""
        all_sequences = {}
        for composer in self.composers:
            all_sequences[composer] = self.get_composer_note_sequences(composer)
        return all_sequences

    def create_midi(self, notes, durations, output_file, tempo=500000):
        """Create a minimal MIDI file from note/duration pairs."""
        # create new MIDI file and track
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)
        
        # set instrument to piano
        track.append(Message('program_change', program=0, time=0))
        
        # set tempo meta-message
        tempo_msg = mido.MetaMessage('set_tempo', tempo=tempo)
        track.append(tempo_msg)
        
        current_time = 0
        timeline = []
        
        # build timeline of note_on and note_off events
        for note, duration in zip(notes, durations):
            if duration < 0:
                # negative duration is a rest: just advance the time
                current_time += abs(duration)
            else:
                # schedule note_on at the current_time
                timeline.append((current_time, 'note_on', note, 64))
                
                # schedule note_off after 'duration' ticks
                end_time = current_time + duration
                timeline.append((end_time, 'note_off', note, 0))
                
                # advance current_time by this duration
                current_time += duration
        
        # sort events by time
        timeline.sort(key=lambda x: x[0])
        
        last_time = 0
        # convert timeline into MIDI messages
        for event in timeline:
            event_time, event_type, note, velocity = event
            delta = event_time - last_time  # time since last event
            
            if event_type == 'note_on':
                track.append(Message('note_on', note=note, velocity=velocity, time=delta))
            elif event_type == 'note_off':
                track.append(Message('note_off', note=note, velocity=velocity, time=delta))
            
            last_time = event_time
        
        mid.save(output_file)

class PCAAnalyzer:
    def __init__(self, n_components=2):
        """Class to perform PCA analysis on transition matrices."""
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        
    def create_transition_matrix(self, model, composer_sequences):
        """Create a transition matrix for each composer by aggregating all their pieces."""
        all_possible_notes = set() # collect all unique notes across all composers
        
        for composer, sequences in composer_sequences.items():
            for piece, notes in sequences.items():
                all_possible_notes.update(notes)
        
        all_possible_notes = sorted(list(all_possible_notes))
        note_to_idx = {note: i for i, note in enumerate(all_possible_notes)}
        
        composer_matrices = {}
        
        for composer, sequences in composer_sequences.items():
            matrix = np.zeros((len(all_possible_notes), len(all_possible_notes))) # init matrix

            aggregated_transitions = defaultdict(Counter) # aggregate transitions from all pieces by this composer
            
            for piece, notes in sequences.items(): # process each piece by this composer
                if len(notes) <= model.order:
                    continue
                
                for i in range(len(notes) - model.order): # process transitions in this piece
                    state = tuple(notes[i:i + model.order])
                    next_note = notes[i + model.order]
                    
                    from_note = state[-1] # only consider notes that are in our filtered set
                    if from_note in note_to_idx and next_note in note_to_idx:
                        aggregated_transitions[from_note][next_note] += 1
            
            for from_note, transitions in aggregated_transitions.items(): # fill matrix with probabilities
                from_idx = note_to_idx[from_note]
                total = sum(transitions.values())
                
                if total == 0:
                    continue
                    
                for to_note, count in transitions.items():
                    to_idx = note_to_idx[to_note]
                    matrix[from_idx, to_idx] = count / total
            
            composer_matrices[composer] = matrix
            
        return composer_matrices
        
    def fit_transform(self, transition_matrices):
        """Apply PCA to transition matrices and return dictionary mapping composers to their PCA-transformed coordinates."""

        matrix_vectors = []
        composers = []
        
        for composer, matrix in transition_matrices.items():
            # Flatten the matrix to a 1D vector for PCA
            matrix_vectors.append(matrix.flatten())
            composers.append(composer)

        X = np.array(matrix_vectors) # convert list of vectors to a 2D NumPy array
        X_scaled = self.scaler.fit_transform(X) # then apply standardization and PCA
        X_pca = self.pca.fit_transform(X_scaled)
        
        return {composer: X_pca[i] for i, composer in enumerate(composers)} # result dict
    
    def plot_pca_results(self, pca_results):
        """Plot PCA results with one point per composer."""
        plt.figure(figsize=(12, 10))
        
        # Assign a distinct color to each composer
        unique_composers = list(pca_results.keys())
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_composers)))
        color_map = {composer: colors[i] for i, composer in enumerate(unique_composers)}
        
        # Plot each composer as a single point
        for composer, coords in pca_results.items():
            plt.scatter(coords[0], coords[1], label=composer, color=color_map[composer], s=100)
            plt.text(coords[0]+0.1, coords[1]+0.1, composer, fontsize=9)
        
        plt.xlabel(f'Principal Component 1 ({self.pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'Principal Component 2 ({self.pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('PCA of Musical Features by Composer')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('composer_pca2.png', dpi=300, bbox_inches='tight')

def visualize_transition_matrix(model, output_filename, max_notes=40):
    """Visualize the Markov transition matrix and save as PNG."""
    # first, collect all unique notes
    all_notes = set()
    for state in model.transitions.keys():
        all_notes.add(state[-1]) # last note in the state
        all_notes.update(model.transitions[state].keys())
    
    all_notes = sorted(list(all_notes))
    
    # limit to max_notes most common notes for readability
    if len(all_notes) > max_notes:
        # count frequency of each note
        note_counts = Counter()
        for state, transitions in model.transitions.items():
            note_counts[state[-1]] += sum(transitions.values())
            for note, count in transitions.items():
                note_counts[note] += count
        
        # keep only the most common notes
        all_notes = [note for note, _ in note_counts.most_common(max_notes)]
        all_notes.sort()
    
    matrix = np.zeros((len(all_notes), len(all_notes))) # empty n x n matrix
    
    # mapping from note to index
    note_to_idx = {note: i for i, note in enumerate(all_notes)}
    
    # aggregate transitions by last note in state
    aggregated_transitions = defaultdict(Counter)
    
    for state, transitions in model.transitions.items():
        last_note = state[-1]
        if last_note in note_to_idx: # only include notes in our filtered set
            for next_note, count in transitions.items():
                if next_note in note_to_idx: 
                    aggregated_transitions[last_note][next_note] += count
    
    # fill the matrix with probabilities
    for from_note, transitions in aggregated_transitions.items():
        from_idx = note_to_idx[from_note]
        total = sum(transitions.values())
        
        if total == 0:
            continue
            
        for to_note, count in transitions.items():
            to_idx = note_to_idx[to_note]
            matrix[from_idx, to_idx] = count / total
    
    # create labels that show note values and corresponding note names
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    labels = [f"{note} ({note_names[note % 12]}{note // 12})" for note in all_notes]
    
    # plot the heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    cax = ax.imshow(matrix, cmap='Blues', aspect='auto')
    cbar = fig.colorbar(cax)
    cbar.set_label("Transition Probability", fontsize=14)
    ax.set_title(f"Markov Transition Matrix (Order = {model.order})", fontsize=16)
    ax.set_xlabel("Next Note", fontsize=14)
    ax.set_ylabel("Current Note", fontsize=14)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')