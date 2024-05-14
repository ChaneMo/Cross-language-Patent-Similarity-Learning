# -*- coding: UTF-8 -*-

# 导入必要的库
import torch
from bert_siamese_similarity import BertSiamese, FGM
from config import config
import pandas as pd
from transformers import BertTokenizer
import scipy.spatial as sp
import random

# 设置随机种子
random.seed(2022)
# 指定对排序结果的前N条进行评估
MRR_NUMBERS = 10
# 模型保存路径
save_model_path = config["save_model_path"]


# 定义模型输入数据获取函数
def get_inputs(texts, mode='single'):
    '''
    :param texts: 输入文本列表
    :param mode: 单条文本还是多条文本输入，默认为单条输入
    :return: 模型输入数据
    '''
    tokenizer = BertTokenizer.from_pretrained(config["bert_name"])
    input_ids, token_type_ids, attention_mask = [], [], []
    # 处理批量输入的文本
    if mode=='batch':
        for i, text in enumerate(texts):
            # print("Processing No.{} sample".format(i))
            res = tokenizer.encode_plus(text, add_special_tokens=True, truncation=True,
                                        max_length=config["max_seq_len"], padding='max_length')
            input_ids.append(torch.tensor([res['input_ids']]))
            token_type_ids.append(torch.tensor([res['token_type_ids']]))
            attention_mask.append(torch.tensor([res['attention_mask']]))
        return input_ids, attention_mask, token_type_ids
    # 处理单条文本
    elif mode=='single':
        res = tokenizer.encode_plus(texts, add_special_tokens=True, truncation=True,
                                    max_length=config["max_seq_len"], padding='max_length')
        return torch.tensor([res['input_ids']]), torch.tensor([res['attention_mask']]), torch.tensor([res['token_type_ids']])


# 定义召回函数
def get_recall_results(model, query, target_text, numbers, embeddings):
    '''
    :param model: 编码模型
    :param query: 查询Query
    :param target_text: 用于构建检索数据库的数据列表
    :param numbers: 使用前N个检索结果进行召回
    :return: 召回文本列表
    '''

    ''' query编码。由于query可变，必须即时编码'''
    with torch.no_grad():
        # 对单条Query进行编码并获取输出的文本向量
        query_ids, query_mask, query_type = get_inputs(query, mode='single')
        model_embedding = model(query_ids, query_mask, query_type)
        query_embedding = list(model_embedding[0])

    ''' 计算query与语料库句子的相似度，输出最相似的前n个'''
    number_top_matches = numbers
    distances = sp.distance.cdist([query_embedding], embeddings, "cosine")[0]
    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])
    abs_results = []
    # 获取排序后的前N条结果
    for idx, distance in results[0:number_top_matches]:
        abs_results.append(target_text[idx].strip())
    # print('Returning results!')

    return abs_results


# 定义排序评估函数
def get_rank(model, query, most_relavent_sent, all_sents, numbers, target_dic_text, target_all_dic_text, single_emb, batch_emb):
    '''
    :param model: 编码模型
    :param query: 数据查询Query
    :param most_relavent_sent: 与Query最相关的期望Target，对应org_dic的value
    :param all_sents: 与Query最相关的期望Target列表，对应org_all_dic的value
    :param numbers: 对排序结果的前N条进行评估
    :param target_dic_text: 用于构建检索向量语料库的Target原文列表，单语Target
    :param target_all_dic_text: 用于构建检索向量语料库的Target原文列表，多语、多个Target
    :return: MRR、MAP值
    '''

    # 1、单条Target召回
    abs_results = get_recall_results(model, query, target_dic_text, numbers, single_emb)

    # 统计MRR
    most_relavent_idx = 0
    if most_relavent_sent in abs_results:
        most_relavent_idx = abs_results.index(most_relavent_sent)
        Mrr = 1 / (most_relavent_idx + 1)
    else:
        # 若前N条没找到最相关的Target，则MRR为0
        Mrr = most_relavent_idx


    # 2、多条Target召回
    abs_results = get_recall_results(model, query, target_all_dic_text, numbers, batch_emb)

    # 统计MAP
    Map = []
    for i, sent in enumerate(all_sents):
        if sent in abs_results:
            sent_idx = abs_results.index(sent)
            # print(sent_idx)
            Map.append(float((i + 1) / (sent_idx + 1)))
        elif sent not in abs_results:
            Map.append(float(0.0))

    # print(Map)

    return Mrr, float(sum(Map) / len(Map))


# 初始化模型
model = BertSiamese()
# 加载训练好的模型权重
model.load_state_dict(torch.load(save_model_path))
# 设置评估模型
model.eval()
# 获取模型评估数据集
data = pd.read_excel('classification_testset.xlsx')
cn_abs = data['abs_Chinese'].tolist()
en_abs = data['abs_English'].tolist()
org_abs = data['abs_Original'].tolist()

# 以小语种原文为Query，中文或者中文&英文为Target
org_dic = {org: cn for org, cn in zip(org_abs, cn_abs)}
org_all_dic = {org: [cn, en] for org, cn, en in zip(org_abs, cn_abs, en_abs)}

# 以中文为Query，小语种或者小语种&英文为Target
cn_dic = {cn: org for cn, org in zip(cn_abs, org_abs)}
cn_all_dic = {cn: [org, en] for cn, org, en in zip(cn_abs, org_abs, en_abs)}

# 以英文为Query，小语种或者小语种&中文为Target
en_dic = {en: org for en, org in zip(en_abs, org_abs)}
en_all_dic = {en: [org, cn] for en, org, cn in zip(en_abs, org_abs, cn_abs)}


# 分别对原文0、中文1、英文2的Query进行测试
all_test = [[org_dic, org_all_dic], [cn_dic, cn_all_dic], [en_dic, en_all_dic]]
round = 0
for dic, all_dic in all_test:
    # 初始化MRR、MAP列表
    MRR = []
    MAP = []
    best_mrr = 0.0
    best_map = 0.0

    # 分别定义MRR、MAP检索数据库的数据
    if round==0:
        target_single = cn_abs
        target_batch = cn_abs + en_abs
    elif round==1:
        target_single = org_abs
        target_batch = org_abs + en_abs
    elif round==2:
        target_single = org_abs
        target_batch = org_abs + cn_abs

    # 1、单条Query
    # 批量获取模型输入数据，用于构建检索数据库
    batch_ids, batch_mask, batch_type = get_inputs(target_single, mode='batch')
    single_embeddings = []
    cnt = 0
    with torch.no_grad():
        # 获取检索语料库向量
        for ids, msk, typ in zip(batch_ids, batch_mask, batch_type):
            if cnt%500==0:
                print("Encoding No.{} sample".format(cnt))
            cnt += 1
            cur_emb = model(ids, msk, typ)
            single_embeddings.append(list(cur_emb[0]))

    # 2、多条Query
    # 批量获取模型输入数据，用于构建检索数据库
    batch_ids, batch_mask, batch_type = get_inputs(target_batch, mode='batch')
    batch_embeddings = []
    cnt = 0
    with torch.no_grad():
        # 获取检索语料库向量
        for ids, msk, typ in zip(batch_ids, batch_mask, batch_type):
            if cnt%500==0:
                print("Encoding No.{} sample".format(cnt))
            cnt += 1
            cur_emb = model(ids, msk, typ)
            batch_embeddings.append(list(cur_emb[0]))

    # 对于评估数据集中的每条数据，单独计算MAP、MRR并存储结果
    for i, query in enumerate(dic):
        if i%200==0:
            print("Calculating No.{} sentence".format(i))
        cur_mrr, cur_map = get_rank(model, query, dic[query], all_dic[query], MRR_NUMBERS, target_single, target_batch, single_embeddings, batch_embeddings)
        MRR.append(cur_mrr)
        MAP.append(cur_map)

    # 输出最终的MRR、MAP结果
    best_mrr = max(best_mrr, sum(MRR)/len(MRR))
    best_map = max(best_map, sum(MAP)/len(MAP))
    if round==0:
        print('Original Query MRR@{}: {}'.format(MRR_NUMBERS, best_mrr))
        print('Original Query MAP@{}: {}'.format(MRR_NUMBERS, best_map))
    elif round==1:
        print('Chinese Query MRR@{}: {}'.format(MRR_NUMBERS, best_mrr))
        print('Chinese Query MAP@{}: {}'.format(MRR_NUMBERS, best_map))
    elif round==2:
        print('English Query MRR@{}: {}'.format(MRR_NUMBERS, best_mrr))
        print('English Query MAP@{}: {}'.format(MRR_NUMBERS, best_map))
    round += 1