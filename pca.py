import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class PCAAnalyzer:
    def __init__(self, n_components=2):
        """
        Class to perform PCA analysis on transition matrices. 
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        
    def fit_transform(self, transition_matrices):
        """
        Apply PCA to transition matrices and return dictionary mapping 
        composers to their PCA-transformed matrices. 
        """
        # Flatten matrices to vectors for PCA
        matrix_vectors = []
        composers = []
        
        for composer, matrix in transition_matrices.items():
            matrix_vectors.append(matrix.flatten())
            composers.append(composer)
                
        # Convert list of vectors to a 2D NumPy array
        X = np.array(matrix_vectors)  # Shape should be (n_composers, n_features)
        
        # Check and reshape if necessary
        if len(X.shape) == 1:
            X = X.reshape(1, -1)  # Reshape to 2D if only one sample
        
        # Apply PCA
        X_scaled = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Store result
        pca_res = {composer: X_pca[i] for i, composer in enumerate(composers)}
        
        return pca_res
    
    def plot_pca_results(self, pca_results):
        """
        Plot PCA results. 
        """
        plt.figure(figsize=(10, 8))
        
        for composer, coords in pca_results.items():
            plt.scatter(coords[0], coords[1], label=composer)
            
        plt.xlabel(f'Principal Component 1 ({self.pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'Principal Component 2 ({self.pca.explained_variance_ratio_[1]:.2%})')
        plt.title('PCA of Musical Features by Composer')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('pca_composers.png')
        plt.show()

def visualize_transition_matrix(model, output_filename, max_notes=40):
    """
    Visualize the Markov transition matrix and save as PNG.
    """
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
    
    # make the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, cmap='Blues', xticklabels=labels, yticklabels=labels)
    
    plt.title(f"Markov Transition Matrix (Order = {model.order})", fontsize=16)
    plt.xlabel("Next Note", fontsize=14)
    plt.ylabel("Current Note", fontsize=14)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
