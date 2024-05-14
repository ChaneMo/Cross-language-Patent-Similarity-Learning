# -*- coding: UTF-8 -*-

# 导入必要的库
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from config import config

# 定义向量标准化函数
def normalize(vecs):
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5

# 孪生神经网络类
class BertSiamese(nn.Module):
    def __init__(self):
        super(BertSiamese, self).__init__()
        bert_name = config["bert_name"]
        bert_config = BertConfig.from_pretrained(bert_name)
        self.bert = BertModel.from_pretrained(bert_name, config=bert_config)
        config_hidden_size = bert_config.hidden_size
        # output_hidden_states设置为True可以获得bert各层的输出
        bert_config.output_hidden_states = True
        config_dropout = bert_config.hidden_dropout_prob
        self.fc = nn.Linear(config_hidden_size, config["hidden"])
        self.drop = nn.Dropout(config_dropout)


    def forward(self, input_ids, attention_mask, token_type_ids):
        '''
        :param input_ids: 将输入到的词映射到模型当中的字典ID
        :param attention_mask: 用于标记subword所处句子和padding的区别，将padding 部分填充为0
        :param token_type_ids: 区分上下句的编码，上句全为0，下句全为1
        :return: bert模型倒数第5层输出的向量
        '''
        # 获取bert模型输出
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        # 获取bert模型各层输出
        hidden_states = outputs[2]
        # 使用bert倒数第五层作为最终的文本向量
        selected_layer = hidden_states[-5]
        sentence_embedding = torch.mean(selected_layer, dim=1)

        return normalize(sentence_embedding)

# 对比损失类
class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        '''
        :param output1: 孪生网络输出的向量1
        :param output2: 孪生网络输出的向量2
        :param label: 真实标签，0表示相似
        :return: 对比损失值
        '''
        euclidean_distance = 1.0-torch.cosine_similarity(output1, output2)
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.), 2)
        )
        return loss_contrastive

# 定义FGM对抗训练方法
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {} # 用于保存模型扰动前的参数

    def attack(
        self,
        epsilon=1.,
        emb_name='bert.embeddings.word_embeddings.weight' # emb_name表示模型中指定对抗训练的参数名
    ):
        '''
        生成扰动和对抗样本
        '''
        for name, param in self.model.named_parameters(): # 遍历模型的所有参数
            if param.requires_grad and emb_name in name: # 只取word embedding层的参数
                self.backup[name] = param.data.clone() # 保存参数值
                norm = torch.norm(param.grad) # 对参数梯度进行二范式归一化
                if norm != 0 and not torch.isnan(norm): # 计算扰动，并在输入参数值上添加扰动
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(
        self,
        emb_name='bert.embeddings.word_embeddings.weight' # emb_name表示模型中指定对抗训练的参数名
    ):
        '''
        恢复添加扰动的参数
        '''
        for name, param in self.model.named_parameters(): # 遍历模型的所有参数
            if param.requires_grad and emb_name in name:  # 只取word embedding层的参数
                assert name in self.backup
                param.data = self.backup[name] # 重新加载保存的参数值
        self.backup = {}

# 定义PGD对抗训练方法
class PGD(object):
    def __init__(self, model, emb_name):
        self.model = model
        self.emb_name = emb_name
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, is_first_attack=False):
        # emb_name参数要换成指定进行对抗训练的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self):
        # emb_name这个参数要指定进行对抗训练的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # print(param.grad)
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]


if __name__=='__main__':
    model = BertSiamese()
    test_input_ids = torch.tensor([[101, 1815, 1747, 1756, 100, 100, 100, 100, 100, 100, 1829,
                                    1825, 1006, 100, 1007, 102, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0],
                                   [101, 5511, 1840, 1820, 100, 1854, 1756, 1981, 100, 2340, 1872,
                                    1015, 100, 100, 100, 100, 1854, 100, 102, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0],
                                   [101, 1016, 9541, 2620, 1840, 1742, 1902, 100, 1854, 1756, 1981,
                                    100, 100, 1910, 100, 100, 100, 1822, 102, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0],
                                   [101, 1848, 100, 1793, 100, 100, 100, 100, 1945, 1945, 100,
                                    1888, 100, 100, 1795, 100, 100, 1931, 100, 100, 100, 100,
                                    1769, 102, 0, 0, 0, 0, 0, 0],
                                   [101, 1639, 100, 1774, 100, 100, 1017, 1640, 100, 1763, 1873,
                                    1741, 100, 100, 100, 100, 100, 102, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0]])
    test_attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0],
                                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0],
                                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0],
                                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                         1, 1, 0, 0, 0, 0, 0, 0],
                                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0]])
    test_token_type_ids = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0]])
    vec = model(test_input_ids, test_attention_mask, test_token_type_ids)
    vec1 = model(test_input_ids, test_attention_mask, test_token_type_ids)
    print(vec==vec1)
    print(torch.cosine_similarity(vec, vec1))
