from dataload import DataLoad
from feature import Encode
from model import NER

import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn import metrics
from transformers import AdamW, get_linear_schedule_with_warmup

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Selection():
    def __init__(self, num_labels, epochs,file_name = 'NCBI-disease'):
        self.epochs = epochs                                      # epochs为训练的次数
        self.num_labels = num_labels                              # num_labels为预测实体的类别
        self.file_name = file_name
        self.encode = Encode(self.file_name,self.train)                      # 进行embedding
        self.model = NER(self.num_labels).to(device)
        self.save_model = os.path.join('./model', self.file_name + '_model.pt')  # 定义保存模型的名称
        self.save_optimizer = os.path.join('./model', self.file_name + '_optimizer.pt')  # 定义保存模型的名称

    def train(self):
        # 进行训练
        train_data = DataLoad(self.file_name, train = True)
        self.model.train()
        optimizer_param = list(self.model.named_parameters())  # named_parameters()获取模型中的参数和参数名字

        """实现L2正则化接口，对模型中的所有参数进行L2正则处理，包括权重w和偏置b"""
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']  # no_decay中存放不进行权重衰减的参数
        # any()函数用于判断给定的可迭代参数iterable是否全部为False，则返回False，如果有一个为True，则返回True
        # 判断optimizer_param中所有的参数。如果不在no_decay中，则进行权重衰减;如果在no_decay中，则不进行权重衰减

        optimizer_grouped_parameters = [
            {'params': [param for name, param in optimizer_param if
                        not any((name in no_decay_name) for no_decay_name in no_decay)], 'weight_decay': 0.01},
            {'params': [param for name, param in optimizer_param if
                        any((name in no_decay_name) for no_decay_name in no_decay)], 'weight_decay': 0.0}
        ]

        # 使用带有权重衰减功能的Adam优化器Adamw
        optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8)
        # 实现学习率预热,optimizer为优化器类,num_warmup_steps为训练多少步进行预热,num_training_steps为总共训练的次数
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=50,
                                                    num_training_steps=len(train_data) * self.epochs)
        # 加载模型参数
        if os.path.exists(self.save_model):
            self.model.load_state_dict(torch.load(self.save_model))
            optimizer.load_state_dict(torch.load(self.save_optimizer))

        # 进行epochs次迭代训练
        for epoch in range(self.epochs):
            bar = tqdm(enumerate(train_data), desc='进行训练', total=len(train_data))
            loss_all = []
            for index, (sentences, targets) in bar:
                input_ids = self.encode.embedding_sentences(sentences)          # 获取input_ids
                labels = self.encode.embedding_targets(targets)                 # 获取labels
                attention_mask = self.encode.attention_mask(input_ids)          # 获取attention_mask
                outputs = self.model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device),
                                     labels=labels.to(device))
                loss, logits = outputs[:2]
                loss_all.append(loss.item())
                """梯度置零，反向传播，参数更新"""
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()  # 更新学习率预热参数
                """输出训练信息"""
                bar.set_description("epoch:{},index:{},loss:{}".format(epoch, index, np.mean(loss_all)))
                # 保存模型
                if index % 10 == 0:
                    torch.save(self.model.state_dict(), self.save_model)
                    torch.save(optimizer.state_dict(), self.save_optimizer)

    def eval(self):
        # 加载模型参数
        if os.path.exists(self.save_model):
            self.model.load_state_dict(torch.load(self.save_model))
        self.model.eval()
        loss_total = 0
        predict_all = []
        label_all = []
        test_data = DataLoad(self.file_name,train = False)
        bar = tqdm(enumerate(test_data), total=len(test_data))
        with torch.no_grad():
            for index, (sentences, targets) in bar:
                input_ids = self.encode.embedding_sentences(sentences)
                labels = self.encode.embedding_targets(targets)
                attention_mask = self.encode.attention_mask(input_ids)
                outputs = self.model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device),
                                     labels=labels.to(device))
                loss, logits = outputs[:2]
                loss_total += loss  # loss_total保存所有损失结果
                labels = labels.data.cpu().numpy()
                predict = torch.argmax(logits, axis= -1).data.cpu().numpy()  # predict为预测标签

                # 修改labels
                labels_length = [len(target) for target in targets]
                labels = [labels[i][1:labels_length[i] + 1] for i in range(len(labels_length))]
                # 修改predict
                predict = [predict[i][1:labels_length[i]+ 1] for i in range(len(labels_length))]

                label_all.extend(labels)
                predict_all.extend(predict)
        with open(os.path.join('./result', self.file_name + '_result.txt'), 'a') as f:
            for i  in range(20):
                result = "\n真实值:"+ str(label_all[i]) +"\n预测值" + str(predict_all[i])
                f.write(result)

        accuracy_score,precision_score,recall_score,f1_score = self.calculate(label_all,predict_all)
        with open(os.path.join('./result',self.file_name + '_result.txt'),'a') as f:
            result = '数据集:'+ str(self.file_name) + '\n准确率为:'+ str(accuracy_score) + '\n精确率为:'+ str(precision_score) + '\n召回率为:'+ str(recall_score) + '\nF1值为:' + str(f1_score) + "\n损失为:" + str(loss_total / len(test_data))
            f.write(result)

        print('数据集:',self.file_name,'\n准确率为:',accuracy_score,'\n精确率为:',precision_score,'\n召回率为:',recall_score ,'\nF1值为:',f1_score,"\n损失为:",loss_total / len(test_data))
        return accuracy_score,precision_score,recall_score,f1_score,loss_total / len(test_data)

    def calculate(self,label_all,predict_all):
        acc_temp = [metrics.accuracy_score(label, predict) for label,predict in zip(label_all,predict_all)]
        precision_temp = [metrics.precision_score(label, predict,average= 'weighted') for label, predict in zip(label_all, predict_all)]
        recall_temp = [metrics.recall_score(label, predict, average='micro') for label, predict in zip(label_all, predict_all)]
        f1_temp = [metrics.f1_score(label, predict,average= 'weighted') for label, predict in zip(label_all, predict_all)]
        return np.mean(acc_temp), np.mean(precision_temp), np.mean(recall_temp), np.mean(f1_temp)

        # from itertools import chain
        # label = list(chain(*label_all))
        # predict = list(chain(*predict_all))
        # acc_temp = metrics.accuracy_score(label, predict)
        # precision_temp = metrics.precision_score(label, predict, average='weighted')
        # recall_temp = metrics.recall_score(label, predict, average='micro')
        # f1_temp = metrics.f1_score(label, predict, average='weighted')
        # return acc_temp, precision_temp, recall_temp, f1_temp

if __name__ == '__main__':
    #   file_name = ['NCBI-disease','BC4CHEMD','BC5CDR-chem']其中之一，选择训练集进行训练
    select = Selection(num_labels=4, epochs=10,file_name= 'BC5CDR-chem')
    # select.train()
    select.eval()
