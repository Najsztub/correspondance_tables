from typing import Any
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE

from alive_progress import alive_bar, alive_it

class PolishSemanticMatcher:
    """Enhanced semantic matcher with specific support for Polish language"""

    POLISH_MODELS = {
        'roberta_large': 'sdadas/mmlw-roberta-large',
        'herbert': 'allegro/herbert-base-cased',
        'multilingual': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        'polish_roberta': 'sdadas/polish-roberta-base-v2'
    }

    def __init__(self, model_type: str = 'herbert'):
        """
        Initialize with specific Polish language model

        Parameters:
        -----------
        model_type : str
            Type of model to use ('herbert', 'multilingual', or 'polish_roberta')
        """
        if model_type not in self.POLISH_MODELS:
            raise ValueError(f"Model type must be one of {list(self.POLISH_MODELS.keys())}")

        self.model_type = model_type
        self.model_name = self.POLISH_MODELS[model_type]
        self.model = SentenceTransformer(self.model_name)

    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for Polish text"""        
        return self.model.encode(text, convert_to_tensor=True, show_progress_bar=False)

    def compute_similarities_tensor(self, emb1: np.ndarray, emb2: np.ndarray) -> Dict[str, float]:
        """Compute multiple similarity metrics between embeddings"""
        return cos_sim(emb1, emb2).detach().item() 


class SimilarityVisualizer:
    """Tools for visualizing similarity matrices and relationships"""

    @staticmethod
    def plot_similarity_heatmap(
        similarity_matrix: np.ndarray,
        source_labels: List[str],
        target_labels: List[str],
        title: str = "Similarity Heatmap"
    ) -> go.Figure:
        """Create interactive heatmap using plotly"""
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=target_labels,
            y=source_labels,
            colorscale='Viridis',
            hoverongaps=False
        ))

        fig.update_layout(
            title=title,
            xaxis_title="COICOP Categories",
            yaxis_title="NACE Categories",
            height=800
        )

        return fig

    @staticmethod
    def plot_embedding_space(
        source_embeddings: np.ndarray,
        target_embeddings: np.ndarray,
        source_labels: List[str],
        target_labels: List[str],
        title: str = "Embedding Space Visualization"
    ) -> go.Figure:
        """Visualize embeddings in 2D space using t-SNE"""
        # Combine embeddings
        combined_embeddings = np.vstack([source_embeddings, target_embeddings])

        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(combined_embeddings)

        # Split back into source and target
        source_2d = embeddings_2d[:len(source_embeddings)]
        target_2d = embeddings_2d[len(source_embeddings):]

        # Create scatter plot
        fig = go.Figure()

        # Add source points
        fig.add_trace(go.Scatter(
            x=source_2d[:, 0],
            y=source_2d[:, 1],
            mode='markers+text',
            name='NACE',
            text=source_labels,
            textposition="top center",
            marker=dict(size=10, color='blue')
        ))

        # Add target points
        fig.add_trace(go.Scatter(
            x=target_2d[:, 0],
            y=target_2d[:, 1],
            mode='markers+text',
            name='COICOP',
            text=target_labels,
            textposition="top center",
            marker=dict(size=10, color='red')
        ))

        fig.update_layout(
            title=title,
            height=800,
            showlegend=True
        )

        return fig

class CorrespondenceAnalyzer:
    """Advanced analysis tools for classification correspondence"""

    @staticmethod
    def analyze_matches(
        similarity_matrix: np.ndarray,
        source_classification: pd.DataFrame,
        target_classification: pd.DataFrame,
        threshold: float = 0.7
    ) -> Dict[str, Any]:
        """Comprehensive analysis of matches and relationships"""

        # Find top matches for each source category
        top_matches = []
        for i, source_row in source_classification.iterrows():
            matches = []
            for j, target_row in target_classification.iterrows():
                similarity = similarity_matrix[i, j]
                if similarity >= threshold:
                    matches.append({
                        'source_code': source_row['code'],
                        'source_desc': source_row['description'],
                        'target_code': target_row['code'],
                        'target_desc': target_row['description'],
                        'similarity': similarity
                    })
            if matches:
                matches.sort(key=lambda x: x['similarity'], reverse=True)
                top_matches.extend(matches)

        # Calculate coverage statistics
        source_coverage = np.mean(np.any(similarity_matrix >= threshold, axis=1))
        target_coverage = np.mean(np.any(similarity_matrix >= threshold, axis=0))

        # Analyze distribution of similarity scores
        all_similarities = similarity_matrix.flatten()
        similarity_stats = {
            'mean': np.mean(all_similarities),
            'median': np.median(all_similarities),
            'std': np.std(all_similarities),
            'min': np.min(all_similarities),
            'max': np.max(all_similarities)
        }

        return {
            'top_matches': pd.DataFrame(top_matches),
            'source_coverage': source_coverage,
            'target_coverage': target_coverage,
            'similarity_stats': similarity_stats
        }

def create_enhanced_polish_correspondence(
    nace_data: pd.DataFrame,
    coicop_data: pd.DataFrame,
    model_type: str = 'herbert'
) -> Dict[str, Any]:
    """
    Create and analyze correspondence between Polish NACE and COICOP classifications
    """
    # Initialize matcher and tools
    matcher = PolishSemanticMatcher(model_type)
    visualizer = SimilarityVisualizer()
    analyzer = CorrespondenceAnalyzer()

    # Generate similarity matrix
    n_source = len(nace_data)
    n_target = len(coicop_data)
    similarity_matrix = np.zeros((n_source, n_target))
    # Compute embeddings for all descriptions
    print("Compute embeddings: source")
    source_embeddings_list = [
        matcher.get_embedding(desc) for desc in alive_it(nace_data['description'])
    ]
    source_embeddings = np.array(source_embeddings_list)
    print("Compute embeddings: target")
    target_embeddings_list = [
        matcher.get_embedding(desc) for desc in alive_it(coicop_data['description'])
    ]
    target_embeddings = np.array(target_embeddings_list)
    # Compute similarity matrix
    print("Compute embeddings: similarity matrix")
    with alive_bar(n_source*n_target) as bar:
        for i in range(n_source):
            for j in range(n_target):
                similarities = matcher.compute_similarities_tensor(
                    source_embeddings_list[i],
                    target_embeddings_list[j]
                )
                similarity_matrix[i, j] = similarities  # Use cosine as primary metric
                bar()

    # Generate visualizations
    heatmap = visualizer.plot_similarity_heatmap(
        similarity_matrix,
        nace_data['code'].tolist(),
        coicop_data['code'].tolist()
    )

    embedding_plot = visualizer.plot_embedding_space(
        source_embeddings,
        target_embeddings,
        nace_data['code'].tolist(),
        coicop_data['code'].tolist()
    )

    # Analyze matches
    analysis = analyzer.analyze_matches(
        similarity_matrix,
        nace_data,
        coicop_data
    )

    return {
        'similarity_matrix': similarity_matrix,
        'visualizations': {
            'heatmap': heatmap,
            'embedding_space': embedding_plot
        },
        'analysis': analysis,
        'source_embeddings': source_embeddings,
        'target_embeddings': target_embeddings
    }


if __name__ == "__main__":
    # Zaczytaj COICOP
    coicop = pd.read_csv(
        "data/processed/kody_COICOP.csv",
        dtype={
            'kod': str,
            'opis': str
        }
    )
    coicop = coicop[['kod', 'opis']].rename(
        columns = {
            'kod': 'code',
            'opis': 'description'
        }
    )
    print(coicop.head())
    
    # Zaczytaj PKWIU
    pkwiu = pd.read_excel(
        'data/raw/pkwiu2015.xls',
        sheet_name = "elementyKlasyfikacji"
    )
    pkwiu = pkwiu[['symbol', 'nazwa']].rename(
            columns = {
                'symbol': 'code',
                'nazwa': 'description'
            })
    # Tylko kategorie bez .0 na końcu
    pkwiu = pkwiu[pkwiu['code'].str.len() <= 8].reset_index(drop=True)
    
    print(pkwiu.head())

    result = create_enhanced_polish_correspondence(
        nace_data = pkwiu,
        coicop_data=coicop,
        model_type="roberta_large"
    )

    # Zapisz macierz zbieżności
    np.save(
        "data/processed/similarity_matrix_embed.npy", 
        result['similarity_matrix']
    )

    # Wyświetl wyniki
    print(result['analysis'])
    result['visualizations']['heatmap'].show()
    result['visualizations']['embedding_space'].show()

    print(result['similarity_matrix'])