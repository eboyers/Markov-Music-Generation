import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import defaultdict, Counter
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pretty_midi
import mido
from mido import MidiFile, MidiTrack, Message
from hmmlearn import hmm
import pickle
import joblib

# Helper functions for MIDI processing
def extract_notes_from_midi(midi_file):
    """Extract notes from a MIDI file and return as a sequence"""
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    notes = []
    
    # Loop through each instrument
    for instrument in midi_data.instruments:

        # Add all notes from this instrument
        for note in instrument.notes:
            notes.append({
                'pitch': note.pitch,
                'start': note.start,
                'end': note.end,
                'duration': note.end - note.start,
                'velocity': note.velocity
            })
    
    # Sort notes by start time
    notes = sorted(notes, key=lambda x: x['start'])
    
    # Extract just the pitches for simplicity
    pitch_sequence = [note['pitch'] for note in notes]
    
    # Also extract timing and duration for more musical generation
    timing_sequence = [(note['duration'], note['velocity']) for note in notes]
    
    return pitch_sequence, timing_sequence, notes

def create_ngrams(sequence, n=1):
    """Create n-grams from a sequence"""
    ngrams = []
    for i in range(len(sequence) - n):
        # Create n-gram as tuple (for hashability)
        if n == 1:
            ngram = (sequence[i],)
            next_item = sequence[i+1]
        else:
            ngram = tuple(sequence[i:i+n])
            next_item = sequence[i+n]
        ngrams.append((ngram, next_item))
    return ngrams

def build_transition_matrix(ngrams, unique_pitches):
    """Build a transition probability matrix from n-grams"""
    matrix_size = len(unique_pitches)
    
    # Create mapping from pitch to index
    pitch_to_idx = {pitch: idx for idx, pitch in enumerate(unique_pitches)}
    
    # Initialize transition matrix with zeros
    transition_matrix = np.zeros((matrix_size, matrix_size))
    
    # Count transitions
    for ngram, next_item in ngrams:
        current_pitch = ngram[-1]  # Use the last note of n-gram as current state
        current_idx = pitch_to_idx[current_pitch]
        next_idx = pitch_to_idx[next_item]
        transition_matrix[current_idx, next_idx] += 1
    
    # Convert counts to probabilities
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    transition_matrix = transition_matrix / row_sums
    
    return transition_matrix, pitch_to_idx

def generate_melody(transition_matrix, pitch_to_idx, unique_pitches, start_pitch=None, length=50):
    """Generate a melody using the transition probability matrix"""
    idx_to_pitch = {idx: pitch for pitch, idx in pitch_to_idx.items()}
    
    # Choose a random starting pitch if none provided
    if start_pitch is None:
        start_pitch = random.choice(unique_pitches)
    
    melody = [start_pitch]
    current_pitch = start_pitch
    
    # Generate the rest of the melody
    for _ in range(length - 1):
        current_idx = pitch_to_idx[current_pitch]
        # Get transition probabilities from current note
        probabilities = transition_matrix[current_idx]
        
        # Choose next note based on probabilities
        next_idx = np.random.choice(len(probabilities), p=probabilities)
        next_pitch = idx_to_pitch[next_idx]
        
        melody.append(next_pitch)
        current_pitch = next_pitch
    
    return melody

def melody_to_midi(melody, output_file, durations=None, velocities=None):
    """Convert a melody (sequence of MIDI pitches) to a MIDI file with optional durations and velocities"""
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)  # Piano
    
    current_time = 0.0
    
    # If no durations provided, use a default
    if durations is None:
        durations = [0.25] * len(melody)
    
    # If no velocities provided, use a default
    if velocities is None:
        velocities = [100] * len(melody)
    
    for i, pitch in enumerate(melody):
        # Get duration and velocity
        duration = durations[i]
        velocity = velocities[i]
        
        # Create a Note object
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=current_time,
            end=current_time + duration
        )
        
        # Add note to piano
        piano.notes.append(note)
        
        # Move time forward
        current_time += duration
    
    # Add piano to the PrettyMIDI object
    midi.instruments.append(piano)
    
    # Write out the MIDI file
    midi.write(output_file)
    return output_file

def extract_features(sequences, composer_labels):
    """Extract musical features for PCA"""
    features = []
    composers = []
    
    for sequence, composer in zip(sequences, composer_labels):
        if len(sequence) < 10:  # Skip very short sequences
            continue
            
        # Calculate basic stats on pitch
        pitch_mean = np.mean(sequence)
        pitch_std = np.std(sequence)
        pitch_range = max(sequence) - min(sequence)
        
        # Calculate pitch intervals
        intervals = [sequence[i+1] - sequence[i] for i in range(len(sequence)-1)]
        interval_mean = np.mean(intervals) if intervals else 0
        interval_std = np.std(intervals) if intervals else 0
        
        # Count occurrence of common intervals
        interval_counts = Counter(intervals)
        step_up = interval_counts.get(1, 0) / max(1, len(intervals))
        step_down = interval_counts.get(-1, 0) / max(1, len(intervals))
        third_up = interval_counts.get(4, 0) / max(1, len(intervals))
        third_down = interval_counts.get(-4, 0) / max(1, len(intervals))
        
        # Calculate note repetition rate
        unique_notes_ratio = len(set(sequence)) / len(sequence)
        
        # Feature vector
        feature_vector = [
            pitch_mean, pitch_std, pitch_range,
            interval_mean, interval_std,
            step_up, step_down, third_up, third_down,
            unique_notes_ratio
        ]
        
        features.append(feature_vector)
        composers.append(composer)
    
    feature_names = [
        'pitch_mean', 'pitch_std', 'pitch_range',
        'interval_mean', 'interval_std',
        'step_up_rate', 'step_down_rate', 'third_up_rate', 'third_down_rate',
        'unique_notes_ratio'
    ]
    
    # Create a DataFrame
    df = pd.DataFrame(features, columns=feature_names)
    df['composer'] = composers
    
    return df

def run_pca(features_df, n_components=2):
    """Run PCA on the extracted features"""
    # Select only numeric features
    X = features_df.drop('composer', axis=1)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X_scaled)
    
    # Create a DataFrame with principal components
    pca_df = pd.DataFrame(
        data=principal_components,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    pca_df['composer'] = features_df['composer']
    
    # Plot PCA results
    plt.figure(figsize=(10, 8))
    composers = pca_df['composer'].unique()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(composers)))
    
    for composer, color in zip(composers, colors):
        composer_data = pca_df[pca_df['composer'] == composer]
        plt.scatter(
            composer_data['PC1'], 
            composer_data['PC2'],
            color=color,
            label=composer,
            alpha=0.7
        )
    
    plt.title('PCA of Musical Features by Composer')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('pca_composers.png')
    
    # Display feature contributions
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=X.columns
    )
    
    return pca_df, loadings, pca

def visualize_transition_matrix(matrix, pitch_to_idx, title="Transition Probability Matrix"):
    """Visualize the transition matrix as a heatmap"""
    idx_to_pitch = {idx: pitch for pitch, idx in pitch_to_idx.items()}
    
    plt.figure(figsize=(12, 10))
    plt.imshow(matrix, cmap='viridis')
    plt.colorbar(label='Transition Probability')
    plt.title(title)
    plt.xlabel('Next Note (MIDI Pitch)')
    plt.ylabel('Current Note (MIDI Pitch)')
    plt.tight_layout()
    plt.savefig('transition_matrix.png')
    
    # Also save a version with a subset of pitches for better readability
    # Focus on the middle range (e.g., piano middle C and two octaves around it)
    middle_range = list(range(60, 85))  # C4 (middle C) to C6
    middle_indices = [pitch_to_idx[p] for p in middle_range if p in pitch_to_idx]
    
    if len(middle_indices) > 5:  # Only create this if we have enough notes in range
        plt.figure(figsize=(12, 10))
        middle_matrix = matrix[np.ix_(middle_indices, middle_indices)]
        plt.imshow(middle_matrix, cmap='viridis')
        plt.colorbar(label='Transition Probability')
        plt.title(title + " (Middle Range)")
        
        # Create readable pitch labels (e.g., "C4", "D4")
        middle_labels = [pretty_midi.note_number_to_name(idx_to_pitch[i]) for i in middle_indices]
        plt.xticks(range(len(middle_labels)), middle_labels, rotation=90)
        plt.yticks(range(len(middle_labels)), middle_labels)
        
        plt.tight_layout()
        plt.savefig('transition_matrix_middle_range.png')

def process_midi_dataset(midi_dir):
    """Process all MIDI files from the dataset directory"""
    print(f"Processing MIDI files from: {midi_dir}")
    
    all_sequences = []
    all_timing_sequences = []
    all_composers = []
    
    # Walk through the directory structure
    for composer in os.listdir(midi_dir):
        composer_path = os.path.join(midi_dir, composer)
        
        if os.path.isdir(composer_path):
            composer_sequence_count = 0
            
            # Process all MIDI files for this composer
            for file in os.listdir(composer_path):
                if file.endswith('.mid') or file.endswith('.midi'):
                    midi_path = os.path.join(composer_path, file)
                    
                    print(f"Processing: {midi_path}")
                    pitch_sequence, timing_sequence, _ = extract_notes_from_midi(midi_path)
                    
                    if pitch_sequence and len(pitch_sequence) > 10:
                        all_sequences.append(pitch_sequence)
                        all_timing_sequences.append(timing_sequence)
                        all_composers.append(composer)
                        composer_sequence_count += 1
            
            print(f"Processed {composer_sequence_count} files for {composer}")
    
    print(f"Total sequences: {len(all_sequences)}")
    print(f"Composers found: {set(all_composers)}")
    
    return all_sequences, all_timing_sequences, all_composers

# New functions for HMM implementation
def prepare_hmm_data(sequences, n_states=10):
    """Prepare data for HMM training"""
    # Flatten all sequences for training
    all_notes = []
    for seq in sequences:
        all_notes.extend(seq)
    
    # Get unique notes and create a mapping
    unique_notes = sorted(set(all_notes))
    note_to_idx = {note: idx for idx, note in enumerate(unique_notes)}
    
    # Convert sequences to arrays of indices
    X = []
    lengths = []
    
    for seq in sequences:
        if len(seq) > 0:
            seq_indices = np.array([note_to_idx[note] for note in seq]).reshape(-1, 1)
            X.append(seq_indices)
            lengths.append(len(seq))
    
    # Concatenate all sequences
    if X:
        X = np.vstack(X)
    else:
        X = np.array([]).reshape(0, 1)
    
    return X, lengths, note_to_idx, unique_notes

def train_hmm_model(all_sequences, n_states=8, n_iter=100):
    """Train a Hidden Markov Model on the music data"""
    print(f"Training HMM with {n_states} hidden states...")
    
    # Prepare data for HMM
    X, lengths, note_to_idx, unique_notes = prepare_hmm_data(all_sequences, n_states)
    
    if X.size == 0:
        print("Error: No data available for HMM training")
        return None, note_to_idx, unique_notes
    
    # Initialize and train the HMM
    model = hmm.MultinomialHMM(n_components=n_states, n_iter=n_iter)
    model.fit(X, lengths)
    
    print(f"HMM training complete. Converged: {model.monitor_.converged}")
    
    return model, note_to_idx, unique_notes

def generate_hmm_melody(model, note_to_idx, unique_notes, length=50, start_note=None):
    """Generate a melody using the trained HMM model"""
    idx_to_note = {idx: note for note, idx in note_to_idx.items()}
    n_notes = len(unique_notes)
    
    # Initialize with a random or specified starting state
    if start_note is not None and start_note in note_to_idx:
        # Use specified start note, but let the model decide the hidden state
        obs = np.array([[note_to_idx[start_note]]])
        _, state = model.decode(obs)
        state = state[0]
    else:
        # Random initial state
        state = np.random.choice(model.n_components)
    
    # Generate the sequence
    obs_sequence = []
    for _ in range(length):
        # Sample from the emission distribution of the current state
        emission_probs = model.emissionprob_[state]
        note_idx = np.random.choice(n_notes, p=emission_probs)
        obs_sequence.append(idx_to_note[note_idx])
        
        # Transition to the next state based on transition matrix
        state = np.random.choice(model.n_components, p=model.transmat_[state])
    
    return obs_sequence

def generate_constrained_hmm_melody(model, note_to_idx, unique_notes, length=50, scale=None, start_note=None):
    """Generate a melody using the trained HMM model but constrain to a musical scale"""
    # Define common scales (in semitone steps from the root)
    scales = {
        'major': [0, 2, 4, 5, 7, 9, 11],
        'minor': [0, 2, 3, 5, 7, 8, 10],
        'pentatonic': [0, 2, 4, 7, 9],
        'blues': [0, 3, 5, 6, 7, 10]
    }
    
    # If no scale specified, default to C major
    if scale is None:
        scale_type = 'major'
        root_note = 60  # C4
    elif isinstance(scale, tuple) and len(scale) == 2:
        scale_type, root_note = scale
    elif isinstance(scale, str) and scale in scales:
        scale_type = scale
        root_note = 60  # C4
    else:
        # Default to C major if invalid scale
        scale_type = 'major'
        root_note = 60
        
    # Create set of allowed pitches based on the scale
    scale_steps = scales.get(scale_type, scales['major'])
    allowed_pitches = set()
    
    # Add notes from several octaves
    for octave in range(-1, 3):  # Roughly from C3 to B5
        for step in scale_steps:
            pitch = root_note + step + (12 * octave)
            if 36 <= pitch <= 84:  # Keep within a reasonable range
                allowed_pitches.add(pitch)
    
    idx_to_note = {idx: note for note, idx in note_to_idx.items()}
    n_notes = len(unique_notes)
    
    # Initialize with a random or specified starting state
    if start_note is not None and start_note in note_to_idx:
        # Use specified start note, but let the model decide the hidden state
        obs = np.array([[note_to_idx[start_note]]])
        _, state = model.decode(obs)
        state = state[0]
    else:
        # Random initial state
        state = np.random.choice(model.n_components)
    
    # Generate the sequence
    obs_sequence = []
    for _ in range(length):
        # Sample from the emission distribution of the current state
        attempt = 0
        max_attempts = 5
        
        while attempt < max_attempts:
            emission_probs = model.emissionprob_[state]
            note_idx = np.random.choice(n_notes, p=emission_probs)
            note = idx_to_note[note_idx]
            
            # If the note is in our allowed scale, accept it
            if note in allowed_pitches:
                obs_sequence.append(note)
                break
            attempt += 1
        
        # If we couldn't find a note in scale after max attempts, take the closest one
        if attempt == max_attempts:
            note = min(allowed_pitches, key=lambda x: abs(x - note))
            obs_sequence.append(note)
        
        # Transition to the next state based on transition matrix
        state = np.random.choice(model.n_components, p=model.transmat_[state])
    
    return obs_sequence

def apply_musical_constraints(melody, key='C', scale_type='major'):
    """Apply musical constraints to make the melody more harmonically coherent"""
    # Define scale degrees for common scales
    scales = {
        'major': [0, 2, 4, 5, 7, 9, 11],
        'minor': [0, 2, 3, 5, 7, 8, 10],
        'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
        'pentatonic': [0, 2, 4, 7, 9]
    }
    
    # Map key names to MIDI note numbers (C4 = 60)
    key_map = {
        'C': 60, 'C#': 61, 'Db': 61, 'D': 62, 'D#': 63, 'Eb': 63,
        'E': 64, 'F': 65, 'F#': 66, 'Gb': 66, 'G': 67, 'G#': 68,
        'Ab': 68, 'A': 69, 'A#': 70, 'Bb': 70, 'B': 71
    }
    
    # Get the key's root note
    root = key_map.get(key, 60)  # Default to C if key not found
    
    # Get the scale pattern
    scale_pattern = scales.get(scale_type, scales['major'])
    
    # Build the complete scale across multiple octaves
    full_scale = []
    for octave in range(-1, 3):  # Generate notes from roughly C3 to B5
        for step in scale_pattern:
            note = root + step + (12 * octave)
            if 36 <= note <= 84:  # Keep within a reasonable range
                full_scale.append(note)
    
    # Map each note to the closest note in the scale
    corrected_melody = []
    for note in melody:
        closest_scale_note = min(full_scale, key=lambda x: abs(x - note))
        corrected_melody.append(closest_scale_note)
    
    return corrected_melody

def main():
    # Directory containing MIDI files - adjust this to your path
    midi_dir = "classical_midis"
    
    # Process all MIDI files
    all_sequences, all_timing_sequences, all_composers = process_midi_dataset(midi_dir)
    
    if not all_sequences:
        print("No valid MIDI sequences found. Please check your dataset directory.")
        return
    
    # Extract all unique pitches from the sequences
    all_pitches = []
    for sequence in all_sequences:
        all_pitches.extend(sequence)
    unique_pitches = sorted(set(all_pitches))
    print(f"Found {len(unique_pitches)} unique MIDI pitches")
    
    # Create features for PCA
    features_df = extract_features(all_sequences, all_composers)
    
    # Run PCA
    pca_df, loadings, pca_model = run_pca(features_df, n_components=3)
    
    print("\nPCA Results:")
    print(f"Explained variance ratios: {pca_model.explained_variance_ratio_}")
    print("\nFeature loadings (contribution to principal components):")
    print(loadings)
    print("\nSaved PCA visualization to 'pca_composers.png'")
    
    # Create n-grams from all sequences
    ngrams = []
    for sequence in all_sequences:
        ngrams.extend(create_ngrams(sequence, n=1))
    
    print(f"Created {len(ngrams)} n-grams for transition matrix")
    
    # Build transition matrix
    transition_matrix, pitch_to_idx = build_transition_matrix(ngrams, unique_pitches)
    
    # Visualize transition matrix
    visualize_transition_matrix(transition_matrix, pitch_to_idx)
    print("Saved transition matrix visualization to 'transition_matrix.png'")
    
    # Train HMM model
    hmm_model, hmm_note_to_idx, hmm_unique_notes = train_hmm_model(all_sequences, n_states=12, n_iter=100)
    
    # Save the trained model for future use
    if hmm_model is not None:
        joblib.dump(hmm_model, 'hmm_model.pkl')
        with open('hmm_mappings.pkl', 'wb') as f:
            pickle.dump((hmm_note_to_idx, hmm_unique_notes), f)
        print("Saved HMM model and mappings")
    
    # Generate melodies: 2 with simple Markov chains and 2 with HMM
    print("\nGenerating melodies...")
    
    # Define common scales for more harmonious melodies
    scales = [
        ('C', 'major'),
        ('A', 'minor'),
        ('G', 'major'),
        ('E', 'minor')
    ]
    
    for i in range(4):
        key, scale_type = scales[i % len(scales)]
        
        if i < 2:  # Generate with Markov chain but apply harmonic constraints
            print(f"\nGenerating melody {i+1} with Markov Chain (constrained to {key} {scale_type})")
            
            # Choose a random starting pitch from the middle range
            middle_range = [p for p in unique_pitches if 60 <= p <= 76]  # C4 to E5
            if not middle_range:
                middle_range = unique_pitches
            
            start_pitch = random.choice(middle_range)
            melody = generate_melody(transition_matrix, pitch_to_idx, unique_pitches, start_pitch, length=50)
            
            # Apply musical constraints to make more harmonious
            melody = apply_musical_constraints(melody, key=key, scale_type=scale_type)
            
            # Generate appropriate durations and velocities
            durations = [0.25 + (random.random() * 0.25) for _ in range(len(melody))]
            velocities = [80 + int(random.random() * 30) for _ in range(len(melody))]
            
            # Convert to MIDI
            output_file = f"generated_melody_markov_{i+1}.mid"
            melody_to_midi(melody, output_file, durations, velocities)
            print(f"Generated melody {i+1}: {output_file}")
            
            # Print the first 10 notes of the melody
            note_names = [pretty_midi.note_number_to_name(note) for note in melody[:10]]
            print(f"First 10 notes: {note_names}")
        
        else:  # Generate with HMM
            if hmm_model is not None:
                print(f"\nGenerating melody {i+1} with HMM (constrained to {key} {scale_type})")
                
                # Convert key notation to MIDI root note
                key_map = {'C': 60, 'A': 69, 'G': 67, 'E': 64}
                root_note = key_map.get(key, 60)
                
                # Generate melody with scale constraints
                melody = generate_constrained_hmm_melody(
                    hmm_model, 
                    hmm_note_to_idx,
                    hmm_unique_notes,
                    length=50,
                    scale=(scale_type, root_note)
                )
                
                # Add some musical variations in timing and velocity
                durations = [0.25 + (random.random() * 0.25) for _ in range(len(melody))]
                velocities = [80 + int(random.random() * 30) for _ in range(len(melody))]
                
                # Convert to MIDI
                output_file = f"generated_melody_hmm_{i-1}.mid"
                melody_to_midi(melody, output_file, durations, velocities)
                print(f"Generated melody {i+1}: {output_file}")
                
                # Print the first 10 notes of the melody
                note_names = [pretty_midi.note_number_to_name(note) for note in melody[:10]]
                print(f"First 10 notes: {note_names}")
    
    # Print detailed summary
    print("\nProject Summary:")
    print(f"- Analyzed {len(all_sequences)} pieces from {len(set(all_composers))} composers")
    print(f"- Built a transition matrix of size {transition_matrix.shape}")
    print(f"- Extracted {features_df.shape[1]-1} musical features for PCA analysis")
    print(f"- Trained an HMM model with {hmm_model.n_components if hmm_model else 0} hidden states")
    print(f"- Generated 2 Markov chain-based melodies and 2 HMM-based melodies")
    print(f"- All melodies are constrained to musical scales to reduce dissonance")

if __name__ == "__main__":
    main()