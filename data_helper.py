# -*- coding: UTF-8 -*-

# 载入必要的库
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from config import config
import pandas as pd


# 定义数据加载类方法
class SiameseDataSet(Dataset):
    def __init__(self, data_path):
        '''
        :param data_path: 数据文件路劲，csv格式文件
        '''
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_name"])

        # 读取csv数据
        data = pd.read_csv(data_path)
        text1 = [str(li) for li in data['sentence_A'].tolist()]
        text2 = [str(li) for li in data['sentence_B'].tolist()]
        label = [int(li) for li in data['label'].tolist()]

        # 生成模型输出数据
        self.input_ids_1, self.attention_mask_1, self.token_type_ids_1 = self._encoder(text1)
        self.input_ids_2, self.attention_mask_2, self.token_type_ids_2 = self._encoder(text2)
        self.label = torch.tensor(label, dtype=torch.int)

    def _encoder(self, texts):
        '''
        :param texts: 文本列表
        :return: 文本列表对应的bert模型输入数据
        '''
        input_ids, token_type_ids, attention_mask = [], [], []
        # 迭代输入文本列表，生成对应的bert输出数据
        for text in texts:
            res = self.tokenizer.encode_plus(text, add_special_tokens=True, truncation=True, max_length=config["max_seq_len"], padding='max_length')
            input_ids.append(res['input_ids'])
            token_type_ids.append(res['token_type_ids'])
            attention_mask.append(res['attention_mask'])
        return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(token_type_ids)


    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        return self.input_ids_1[item], self.attention_mask_1[item], self.token_type_ids_1[item], \
               self.input_ids_2[item], self.attention_mask_2[item], self.token_type_ids_2[item], self.label[item]
