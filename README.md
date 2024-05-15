# Cross-language Patent Similarity Learning
## 1.模型简介
&emsp;&emsp;基于孪生神经网络并融入对抗训练方法构建了一个孪生对抗神经网络框架。该框架面向多语言、跨语种的专利文本表示模型训练任务及检索应用任务所构建，采用对比损失（Contrastive Loss）作为模型训练的损失函数，旨在更好地微调训练相关文本表示模型。孪生对抗神经网络框架及微调训练后的模型有效性主要通过在自建的泰语、越南语小语种专利平行语料以及自建的包含泰语、越南语、德语、法语、日语、韩语、俄语 7 种语言在内的专利平行语料上设计多组对比实验进行验证。
![1715782055563](https://github.com/ChaneMo/sentence_similarity_learning/assets/91654630/3490949a-4992-40a7-974b-d1c0bdd8766e)
## 2.跨语种数据训练结果对比
### 2.1 模型检索性能评估
&emsp;&emsp;通过在包含同一专利小语种原文及中文翻译版本的数据库中，以小语种文本表示为检索对象检索最相近的N个文本表示，计算对应中文翻译版本文本表示所在排位得出指标。
![1715782094738](https://github.com/ChaneMo/sentence_similarity_learning/assets/91654630/41f53341-f4c2-4074-94e7-30323629c0d7)
### 2.2 模型表示能力示意
&emsp;&emsp;同一专利分别包含中文、英文、小语种原文三个版本的文本表示，图中编号数字相同的点代表同一个专利。
![1715781949419](https://github.com/ChaneMo/sentence_similarity_learning/assets/91654630/84092bb9-52fe-4199-a9a1-dc45a62a3c52)
![1715782023457](https://github.com/ChaneMo/sentence_similarity_learning/assets/91654630/d7650309-ad93-46b9-ae96-253ab4501543)
