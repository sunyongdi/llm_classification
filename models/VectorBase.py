# -*- coding: utf-8 -*-
'''
@File    :   VectorBase.py
@Time    :   2024/08/13 15:09:41
@Author  :   sunyd 
@Email   :   sunyongdi@outlook.com 
@Version :   1.0
@Desc    :   None
'''
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
from typing import List, Dict
from .Embeddings import BaseEmbeddings

class VectorStore:
    def __init__(self, EmbeddingModel: BaseEmbeddings, db_path: str='milvus_demo.db', collection_name: str='my_rag_collection') -> None:
        self.EmbeddingModel = EmbeddingModel
        self.milvus_client = MilvusClient(uri=db_path)
        self.collection_name = collection_name

    def create_collection(self) -> None:
        if self.milvus_client.has_collection(self.collection_name):
            self.milvus_client.drop_collection(self.collection_name)

        # # 创建集合的schema
        # fields = [
        #     FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        #     FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.EmbeddingModel._model.config.hidden_size),
        #     FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512)
        # ]
        # collection_schema = CollectionSchema(fields, self.collection_name)

        self.milvus_client.create_collection(
            collection_name=self.collection_name,
            # schema=collection_schema,
            dimension=self.EmbeddingModel._model.config.hidden_size,
            metric_type="IP",  # Inner product distance
            consistency_level="Strong",  # Strong consistency level
        )
    
    def insert(self, data: List[dict]) -> None:
        self.milvus_client.insert(collection_name=self.collection_name, data=data)

    def query(self, query: str, k: int = 3) -> List[str]:
        search_res = self.milvus_client.search(
            collection_name=self.collection_name,
            data=[
                self.EmbeddingModel.get_embedding(query)
            ],
            limit=k,  # Return top 3 results
            # search_params={"metric_type": "IP", "params": {}},  # Inner product distance
            output_fields=["text"],  # Return the text field
        )

        return [(res["entity"]["text"], res["distance"]) for res in search_res[0]]

if __name__ == "__main__":
    from Embeddings import BgeEmbedding
    model_path = '/root/sunyd/model_hub/ZhipuAI/bge-large-zh-v1___5'
    model = BgeEmbedding(model_path)
    vec_model = VectorStore(model, db_path='/root/sunyd/llms/llm_classification/storage/rag.db')
    vec_model.create_collection()
    data = []
    
    for i, line in enumerate(['你好', 'hello']):
        data.append({"id": i, "vector": model.get_embedding(line), "text": line})
    vec_model.insert(data)
    res = vec_model.query('你好')
    print(res)


