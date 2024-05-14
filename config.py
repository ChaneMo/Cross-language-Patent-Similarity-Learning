# -*- coding: UTF-8 -*-

# 定义训练的参数
config = {
    # 指定调用的模型权重版本
    "bert_name": "bert-base-multilingual-cased",
    # 指定隐藏层大小
    "hidden": 768,
    # 指定最长文本长度
    "max_seq_len": 512,
    # 指定batch size
    "batch_size": 32,
    # 指定学习率
    "lr": 5e-5,
    # 指定训练次数
    "epochs": 5,
    # 指定模型保存路劲
    "save_model_path": "./model/siamesebert.pth"
}
