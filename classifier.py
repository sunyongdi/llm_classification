# -*- coding: utf-8 -*-
'''
@File    :   classifier.py
@Time    :   2024/08/13 10:27:11
@Author  :   sunyd 
@Email   :   sunyongdi@outlook.com 
@Version :   1.0
@Desc    :   None
'''
from models.LLM import QwenMode
from models.VectorBase import VectorStore
from models.Embeddings import BgeEmbedding
from config import PROMPT_TEMPLATE, EMBEDDING_PATH, DB_PATH, LLM_PATH
from loguru import logger



class VecLlmClassifier:
    def __init__(self) -> None:
        self.emb_model = BgeEmbedding(EMBEDDING_PATH)
        self.retrieval = VectorStore(self.emb_model, DB_PATH)
        self.llm = QwenMode(LLM_PATH)


    def predict(self, query: str, icl=True) -> str:
        task_description = PROMPT_TEMPLATE['CLASSIFY_PROMPT_TEMPLATE']
        demonstrations = ''
        # ICL
        if icl:
            demonstrations = self.retrieval.query(query, k=3)
            logger.info('大模型生成解释........')
            demonstrations = ['文本：' + demonstration[0] + '\n' + '原因:' + \
                            self.llm.chat([{'role': 'user', 'content': PROMPT_TEMPLATE['REASON_PROMPT_TEMPLATE'].format(content=demonstration[0])}]) \
                                for demonstration in demonstrations]
        
        
        # LLM
        logger.info('大模型进行推理........')
        output = self.llm.chat([{
            'role': 'user', 
            'content': task_description.format(examples=demonstrations, options='财经、彩票、房产、股票、家居、教育、科技、社会、时尚、时政、体育、星座、游戏、娱乐', options_detail = '', query=query)}])
        return output
    
    def predict1(self, query: str) -> str:
        """已搜代分"""
        output = self.retrieval.query(query, k=1)
        return output[0][0][-2:]
    
if __name__ == '__main__':
    vlc = VecLlmClassifier()
    res = vlc.predict1('《神兆OL》剑灵归心新人大奖即将揭晓眼看剑灵归心区开放已接近一个月，我是职业之王活动已接近尾声，究竟鹿死谁手还是未知数，玩家的排行形式尚不明朗，很多玩家还有机会，下面是活动的详细内容，看看你到底排第几？能拿到什么东西？活动名称：我是职业之王活动时间：12月3日—1月3日活动规则：在新区开放的同时将同时开放世界职业排行榜，分为7个职业：力士，拳师，侠客，盗贼，猎人，术士，道士。排行标准为单一职业的等级排行，该排行榜一天更新一次，活动时间持续一个月，一个月后即1月3日活动结束时，按照世界职业排行榜情况给予不同的奖励。第一名：获得御赐金箱20个，将之魂石5颗，龙王之鳞1个(一周版)，踏焰白玉兽(一月版)，金元宝2000第二名：获得御赐金箱15个，将之魂石4颗，60级踏焰白玉兽(一月版)，金元宝1000第三名：获得御赐金箱10个，将之魂石3颗，60级踏焰白玉兽(一月版)，金元宝500第四名：获得御赐金箱5个，将之魂石2颗，复活药10个，60级踏焰白玉兽(一周版)第五名：获得将之魂石2颗，复活药10个，60级踏焰白玉兽(一周版)第六名：获得将之魂石2颗，复活药10个，60级浴血铁角犀(一周版)第七名：获得将之魂石2颗，复活药10个，二级力量宝石，二级悟性宝石，二级灵巧宝石各一颗第八名：获得将之魂石2颗，复活药10个，二级力量宝石，二级悟性宝石，二级灵巧宝石各一颗第九名：获得将之魂石2颗，复活药10个，大地药水，优雅药水，智慧药水各一瓶第十名：获得将之魂石2颗，复活药10个，大地药水，优雅药水，智慧药水各一瓶活动时间仅剩不到一周时间了，1月3日晚12时即将锁定排行情况，请玩家们继续努力争夺排行名次，为了自己的一片天地奋力打拼吧！')
    print(res)

    
