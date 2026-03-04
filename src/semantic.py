#semantic.py
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

class SemanticAnalyzer:
    def __init__(self, model_name="microsoft/codebert-base", lazy_load=False):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not lazy_load:
            self._load_model()

    def _load_model(self):
        try:
            print(f"Loading semantic model: {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            self.model = None

    def get_embedding(self, code):
        if self.model is None:
            if self.tokenizer is None:
                self._load_model()
            
            if self.model is None:
                return np.zeros(768)
        
        try:
            inputs = self.tokenizer(code, return_tensors="pt", max_length=512, truncation=True, padding=True)
            for k, v in inputs.items():
                inputs[k] = v.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
                
            return embedding
        except Exception as e:
            print(f"Error computing embedding: {e}")
            return np.zeros(768)

    def calculate_similarity(self, code1, code2):
        emb1 = self.get_embedding(code1)
        emb2 = self.get_embedding(code2)
        
        if np.all(emb1 == 0) or np.all(emb2 == 0):
            return 0.0
            
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return float(np.dot(emb1, emb2) / (norm1 * norm2))
