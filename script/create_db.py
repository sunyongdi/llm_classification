from models.VectorBase import VectorStore
from models.Embeddings import BgeEmbedding
from tqdm import tqdm

def process_data():
    pass

def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            label, content = line.split('\t')
            yield '文本：' + content + '上述文本所属的类别是' + label


if __name__ == '__main__':
    file_path = '/root/sunyd/llms/llm_classification/data/cnews/cnews.train.txt'
    emb_path = '/root/sunyd/model_hub/ZhipuAI/bge-large-zh-v1___5'
    emb_model = BgeEmbedding(emb_path)
    vec = VectorStore(emb_model, db_path='/root/sunyd/llms/llm_classification/storage/rag.db')
    # 初始化
    vec.create_collection()
    # 创建
    dataset = read_txt(file_path)
    for i, text in enumerate(tqdm(dataset, desc='解析文本：')):
        vec.insert({"id": i, "vector": emb_model.get_embedding(text), "text": text})
    res = vec.query('马晓旭意外受伤让国奥警惕 无奈大雨格外青睐殷家军记者傅亚雨沈阳报道 来到沈阳，国奥队依然没有摆脱雨水的困扰。', k=3)
    print(res)
    