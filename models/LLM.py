# -*- coding: utf-8 -*-
'''
@File    :   LLM.py
@Time    :   2024/08/13 11:28:57
@Author  :   sunyd 
@Email   :   sunyongdi@outlook.com 
@Version :   1.0
@Desc    :   None
'''

from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, List
from loguru import logger
import torch
from abc import ABC, abstractmethod

def _gc():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

class BaseModel(ABC):

    @abstractmethod
    def chat(self, messages: str, max_new_tokens: int = 512, do_sample:bool = False, top_k: float = 1, temperature: float = 0.8):
        """大模型对话"""


class QwenMode(BaseModel):
    def __init__(self, model_path) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = self.model.eval()
        self.device = self.model.device

        logger.info("load LLM Model done")

    def chat(self, 
             messages: str,
             max_new_tokens: int = 1024,
             do_sample:bool = False,
             top_k: float = 1,
             temperature: float = 0.8
             ) -> str:
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_k=top_k,
            temperature=temperature
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        logger.info(f'input_tokens:{len(model_inputs.input_ids.tolist()[0])} \t generated_ids:{len(generated_ids[0].tolist())} \t all_tokens:{len(model_inputs.input_ids.tolist()[0]) + len(generated_ids[0].tolist())}')
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        _gc()
        return response
    
if __name__ == "__main__":
    LLM_PATH = '/root/sunyd/model_hub/qwen/Qwen2-7B-Instruct'
    llm_model = QwenMode(LLM_PATH)
    messages = [{'role': 'user', 'content': '文本：流感袭击广东主帅打点滴坚持 李春江无惧“失声”李春江无惧“失声”本报天津电\xa0“首发的队员上去后一定要打好开局，尤其是保护好篮板球，还有就是对于对方重点人的防守……”赛前准备会上，李春江用近乎沙哑的嗓音提醒队员。此次客场之旅，卫冕冠军将士遭遇了寒流的袭击。太原客场，先是王仕鹏在毫无征兆的情况下发烧，教练组在赛前更换了大名单。转战天津后，队医为了预防流感进一步蔓延，让每个队员都喝了板蓝根冲剂。但大鹏刚退烧，李春江和刘晓宇又发烧了。刚一到天津，刘晓宇就感觉不适，前天下午就在队医的带领下前往医院打了点滴，为了不影响到其他队员，球队还专门给刘晓宇安排了单间。即便如此，流感还是没有得到遏制。前天晚上，李春江开始发高烧，撑了一夜后于昨天上午前往医院打了点滴，下午刘晓宇第二次前往医院。为了让其得到休息，昨天的比赛晓宇根本没有上场。队员感冒起码有其他队员轮换，但李春江的位置是无人能顶替的，比赛中李春江依旧用沙哑的嗓音向场内队员高喊注意事项。“还行吧，病来如山倒，我也没办法，只能硬撑着了。”赛后走出体育馆时，李春江在西装外套上了厚厚的大衣和棉裤。其实，大鹏的感冒也没有完全好转，昨天虽然退烧了，但浑身上下没有劲。“腿都是软的，这还算好的，主要是嗓子还在发炎，上场打一会就感觉喘不过气来。”赛后王仕鹏告诉记者。虽然感冒尚未痊愈，但大鹏却展现出了其惊人的效率，他在19分钟内砍下了24分和4个篮板。“现在球队的情况大家都了解，我不能因为感冒而退缩，越是这种艰难时刻，我们老队员就更应该站出来。”王仕鹏说。本报宏远随队记者 刘爱琳\n上述文本所属的类别是体育, 请给出对该分类结果的合理解释。'}]
    print(llm_model.chat(messages))