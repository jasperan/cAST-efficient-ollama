import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union

class CodeEmbedder:
    def __init__(self, model_name: str = 'microsoft/codebert-base'):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        if isinstance(text, str):
            text = [text]
        embeddings = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return embeddings
