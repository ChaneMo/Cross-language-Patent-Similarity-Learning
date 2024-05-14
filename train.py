# -*- coding: UTF-8 -*-

# 加载必要的库
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_cosine_schedule_with_warmup
import numpy as np
from bert_siamese_similarity import BertSiamese, ContrastiveLoss, FGM, PGD
from data_helper import SiameseDataSet
from config import config

# 定义必要参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = config["batch_size"]
lr = config["lr"]
epochs = config["epochs"]
save_model_path = config["save_model_path"]
criterion = ContrastiveLoss(0.9)
train_data = "../data/train.csv"
dev_data = "../data/dev.csv"
test_data = "../data/test.csv"


# 定义验证函数
def dev(model, data_loader):
    '''
    :param model: 待验证的模型
    :param data_loader: 验证数据
    :return: 验证结果
    '''
    dev_loss = 0.
    step = 0.
    model.eval()

    # result_tensor = []
    sim_results = []
    diff_results = []

    with torch.no_grad():
        for i, batch in enumerate(data_loader, start=1):
            # batch = [d.to(device) for d in batch]
            batch = [d for d in batch]
            output1 = model(*batch[0:3])
            output2 = model(*batch[3:6])
            label = batch[-1]

            # 计算余弦相似度
            sims = torch.cosine_similarity(output1, output2)
            print(sims, label)
            for i, li in enumerate(sims.tolist()):
                if label[i] == 0:
                    sim_results.append(li)
                else:
                    diff_results.append(li)
            
            # print(torch.cosine_similarity(output1, output2), label)
            loss = criterion(output1, output2, label)
            dev_loss += loss.item()
            step += 1
    
    if not sim_results:
        print("No similarity pairs")
        print("Average different pairs similarity:", sum(diff_results)/len(diff_results))
    elif not diff_results:
        print("Average similar pairs similarity:", sum(sim_results)/len(sim_results))
        print("No different pairs")
    else:
        print("Average similar pairs similarity:", sum(sim_results)/len(sim_results))
        print("Average different pairs similarity:", sum(diff_results)/len(diff_results))

    return dev_loss/step

# 定义训练函数
def train():
    # 加载数据集
    train_dataloader = DataLoader(SiameseDataSet(train_data), batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(SiameseDataSet(dev_data), batch_size=batch_size, shuffle=False)

    # 加载模型，优化器
    model = BertSiamese()
    # unfreeze_layers = ['word_embeddings.', 'bert.pooler', 'out.']
    unfreeze_layers = ['layer.7.']
    for name, param in model.named_parameters():
        print(name, param.size())

    print("*"*30)
    print('\n')

    # 解冻相关训练层
    for name, param in model.named_parameters():
        param.requires_grad = False
        for ele in unfreeze_layers:
            if ele in name:
                param.requires_grad = True
                break

    # 输出待训练各层的情况
    for name, param in model.named_parameters():
        if param.requires_grad:
            # param.retain_grad()
            print(name, param.size())

    # 定义优化器、warmup机制
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataloader),
                                                num_training_steps=epochs * len(train_dataloader))

    # FGM对抗训练
    # fgm = FGM(model)

    # PGD对抗训练
    pgd = PGD(model, emb_name='word_embeddings.')
    # PGD对抗迭代K次
    K=3

    # 开始训练
    best_dev_loss = float('inf')
    for epoch in range(1, epochs+1):
        model.train()
        for i, batch in enumerate(train_dataloader, start=1):
            optimizer.zero_grad()
            loss = criterion(model(*batch[0:3]), model(*batch[3:6]), batch[-1])
            loss.backward()

            # FGM对抗训练
            # fgm.attack()  # （#2）在embedding上添加对抗扰动
            # output1 = model(*batch[0:3])
            # output2 = model(*batch[3:6])
            # label = batch[-1]
            # loss_adv = criterion(output1, output2, label)  # （#3）计算含有扰动的对抗样本的loss
            # loss_adv = criterion(model(*batch[0:3]), model(*batch[3:6]), batch[-1])
            # loss_adv.backward()  # （#4）反向传播，并在正常的grad基础上，累加对抗训练的梯度
            # fgm.restore()  # （#5）恢复embedding参数

            # PGD对抗训练
            pgd.backup_grad()
            for t in range(K):
                pgd.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
                if t != K - 1:
                    model.zero_grad()
                else:
                    pgd.restore_grad()
                loss_adv = criterion(model(*batch[0:3]), model(*batch[3:6]), batch[-1])
                # loss_adv = loss_adv.mean()
                loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            pgd.restore()  # 恢复embedding参数

            optimizer.step()
            scheduler.step()
            model.zero_grad()

            if i % 3 == 0:
                print("Train epoch:{} step:{} loss:{}".format(epoch, i, loss.item()))

        # 验证并保存最佳模型
        dev_loss = dev(model, dev_dataloader)
        print("Dev  epoch:{} loss:{}".format(epoch, dev_loss))
        if dev_loss < best_dev_loss:
            torch.save(model.state_dict(), save_model_path)
            best_dev_loss = dev_loss

if __name__ == "__main__":
    train()
    # model = BertSiamese()
    # model.load_state_dict(torch.load(save_model_path))
    # test_dataloader = DataLoader(SiameseDataSet(test_data), batch_size=batch_size, shuffle=False)
    # dev(model, test_dataloader)
