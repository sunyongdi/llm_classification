from models.VectorBase import VectorStore
from models.Embeddings import BgeEmbedding
from tqdm import tqdm

sample_test = open('/root/sunyd/llms/llm_classification/data/cnews/sample_test.txt', 'a', encoding='utf-8')

def read_txt(file_path, count=200):
    with open(file_path, 'r', encoding='utf-8') as f:
        data_dict = dict()
        for line in f:
            label, content = line.split('\t')
            data_dict.setdefault(label, [])
            if len(data_dict[label]) < count:
                data_dict[label].append(content)

    for k, v in data_dict.items():
        for i in v:
            sample_test.write(f'{k.strip()}\t{i.strip()}\n')
    
    sample_test.close()


if __name__ == '__main__':
    file_path = '/root/sunyd/llms/llm_classification/data/cnews/cnews.test.txt'
    read_txt(file_path)
    # emb_path = '/root/sunyd/model_hub/ZhipuAI/bge-large-zh-v1___5'
    # emb_model = BgeEmbedding(emb_path)
    # vec = VectorStore(emb_model, db_path='/root/sunyd/llms/llm_classification/storage/rag.db')
    # # 初始化
    # vec.create_collection()
    # # 创建
    # dataset = read_txt(file_path)
    # for i, text in enumerate(tqdm(dataset, desc='解析文本：')):
    #     vec.insert({"id": i, "vector": emb_model.get_embedding(text), "text": text})
    # res = vec.query('马晓旭意外受伤让国奥警惕 无奈大雨格外青睐殷家军记者傅亚雨沈阳报道 来到沈阳，国奥队依然没有摆脱雨水的困扰。', k=3)
    # print(res)
    