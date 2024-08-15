# -*- coding: utf-8 -*-
'''
@File    :   Embeddings.py
@Time    :   2024/08/13 14:51:01
@Author  :   sunyd 
@Email   :   sunyongdi@outlook.com 
@Version :   1.0
@Desc    :   None
'''
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from abc import ABC, abstractmethod


class BaseEmbeddings(ABC):
    """
    Base class for embeddings
    """
    @abstractmethod
    def get_embedding(self, text: str, model: str) -> List[float]:
        """获取向量"""

    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        """
        calculate cosine similarity between two vectors
        """
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude


class BgeEmbedding(BaseEmbeddings):
    """
    class for BGE embeddings
    """

    def __init__(self, path: str = 'BAAI/bge-base-zh-v1.5') -> None:
        self._model, self._tokenizer = self.load_model(path)

    def get_embedding(self, text: str) -> List[float]:
        import torch
        encoded_input = self._tokenizer([text], max_length=512, padding='max_length', truncation=True, return_tensors='pt')
        encoded_input = {k: v.to(self._model.device) for k, v in encoded_input.items()}
        with torch.no_grad():
            model_output = self._model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings[0].tolist()

    def load_model(self, path: str):
        import torch
        from transformers import AutoModel, AutoTokenizer
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModel.from_pretrained(path).to(device)
        model.eval()
        return model, tokenizer
    
if __name__ == '__main__':
    model_path = '/root/sunyd/model_hub/ZhipuAI/bge-large-zh-v1___5'
    model = BgeEmbedding(model_path)
    print(model._model.config.hidden_size)
    # emb1 = model.get_embedding('你好')
    # emb2 = model.get_embedding('hello')
    # print(model.cosine_similarity(emb1, emb2))