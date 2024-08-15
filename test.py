from sklearn.metrics import classification_report
from classifier import VecLlmClassifier
from tqdm import tqdm
from loguru import logger

logger.add(sink='runtime_no_icl.log')

labels = ['家居', '体育', '教育', '房产', '科技', '娱乐', '游戏', '财经', '时尚', '时政']

import time

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds to execute.")
        return result
    return wrapper

@timer_decorator
def get_labels(data_path):
    labels = set()
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            true_label, content = line.split('\t')
            labels.add(true_label)
    return list(labels)

def handle_response(llm_response):
    pred_label = None
    for label in labels:
        if label in llm_response:
            pred_label = label
            break
    return pred_label

@timer_decorator
def main(data_path):
    vlc = VecLlmClassifier()
    test_outputs, test_targets = [], []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=10000, desc='预测文本:'):
            true_label, content = line.split('\t')
            test_targets.append(true_label.strip())
            llm_response = vlc.predict1(content.strip())
            # llm_response = vlc.predict(content.strip(), icl=True)

            logger.info(f'预测结果：{llm_response} \t 真实类别：{true_label}')
            pred_label = handle_response(llm_response)
            if pred_label:
                test_outputs.append(pred_label)
            else:
                logger.error('大模型推理错误：{}'.format(llm_response))
                test_outputs.append('家居')
    
    report = classification_report(test_targets, test_outputs, target_names=labels)
    return report


if __name__ == '__main__':
    # test_data_path = '/root/sunyd/llms/llm_classification/data/cnews/sample_test.txt'
    test_data_path = '/root/sunyd/llms/llm_classification/data/cnews/cnews.test.txt'
    print(get_labels(test_data_path))
    print(main(test_data_path))
