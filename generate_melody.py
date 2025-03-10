import os

from utils import *
from model import *

def generate_melody(key_name, scale_type):
    """
    Generates a Markov chain-based melody. 
    """
    key = KEYS[key_name]
    
    # generate the melody
    melody, durations = model.generate_melody(
        key=key,
        scale_type=scale_type,
        phrase_length=16,  # Longer phrases for more cohesive melodies
    )
    
    output_filename = f"{key_name}_{scale_type}.mid"
    output_path = os.path.join(output_dir, output_filename)
    tempo = 480000
    
    # generate the MIDI file
    processor.create_midi(
        notes=melody,
        durations=durations,
        output_file=output_path,
        tempo=tempo
    )

KEYS = {
    'C': 60, 'C#': 61, 'Db': 61, 'D': 62, 'D#': 63, 'Eb': 63, 
    'E': 64, 'F': 65, 'F#': 66, 'Gb': 66, 'G': 67, 'G#': 68, 
    'Ab': 68, 'A': 69, 'A#': 70, 'Bb': 70, 'B': 71
}

output_dir = "new2"
os.makedirs(output_dir, exist_ok=True)

processor = MIDIProcessor()

all_sequences = processor.get_all_note_sequences()

model = MarkovModel(order=5) # model of order 5 to inform decisions based on previous 5 notes
all_note_sequences = []

for composer_sequences in all_sequences.values():
    all_note_sequences.extend(list(composer_sequences.values())) # combine all note sequences for training

# train the model
model.train(all_note_sequences, phrase_length=8)

# see actual Markov transition matrix
# visualize_transition_matrix(model, "transition_matrix.png")

# run PCA
# pca_analyzer = PCAAnalyzer(n_components=2)
# composer_matrices = pca_analyzer.create_transition_matrix(model, all_sequences)
# pca_results = pca_analyzer.fit_transform(composer_matrices)
# pca_analyzer.plot_pca_results(pca_results) 

# generate melodies based on a key and mode; example use
generate_melody('E', 'major')
generate_melody('A', 'harmonic_minor')