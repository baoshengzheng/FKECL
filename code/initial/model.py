#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

from dataloader import TestDataset


class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma,
                 double_entity_embedding=True, double_relation_embedding=False): # 两个布尔值，分别表示是否使用双倍大小的嵌入来表示实体和关系
        super(KGEModel, self).__init__()
        self.model_name = model_name # 'RotatE'
        self.nentity = nentity       # 225
        self.nrelation = nrelation   # 17
        self.hidden_dim = hidden_dim # 64
        self.epsilon = 2.0

        self.gamma = nn.Parameter(   #这个张量是模型中的一个可学习参数。requires_grad=False表示该参数不需要求导。
            torch.Tensor([gamma]),   # 张量[19.9]  # 计算三元组的分数的边界值gamma,初始化范围的计算也用到    fixed margin
            requires_grad=False
        )

        # self.embedding_range = nn.Parameter(
        #     torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
        #     requires_grad=False
        # )                                                                             # (19.9  +  2)/64 = 0.3421874940395355
        self.embedding_range = (self.gamma.item() + self.epsilon) / hidden_dim          # 这行代码计算了嵌入矩阵的范围，即嵌入向量每个维度的范围。这个范围是一个常数，用来控制嵌入矩阵中的值。 0.3421874940395355

        self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim     # 这两行代码根据double_entity_embedding和double_relation_embedding的值计算了实体和关系嵌入向量的维度。
        self.relation_dim = hidden_dim * 2 if double_relation_embedding else hidden_dim #          若为true 则为128 否则为64

        # self.entity_embedding = nn.Embedding(nentity, self.entity_dim)
        # self.relation_embedding = nn.Embedding(nrelation, self.relation_dim)
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))     # entity_embedding [225,128] 全零  ******************************
        nn.init.uniform_(
            tensor=self.entity_embedding, # 这个张量是实体嵌入矩阵。nn.init.uniform_函数使用均匀分布来初始化实体嵌入矩阵中的值  entity_embedding [225,128]  每行都对应了一个实体
            # a=-self.embedding_range.item(),
            # b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))  # relation_embedding [17,64]维度矩阵 代表关系属性 同样每行对应一组关系
        nn.init.uniform_(
            tensor=self.relation_embedding,
            # a=-self.embedding_range.item(),
            # b=self.embedding_range.item()
        )

        if model_name == 'pRotatE':                     # 没啥用 因为没用 pRotatE
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range]]))

        # Do not forget to modify this line when you add a new model in the "forward" function  # 也没啥用一眼就看懂
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE']:
            raise ValueError('model %s not supported' % model_name)

        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')

    def forward(self, sample, mode='single'):  # sample (tensor([[ 68,  16,  ...16, 117]]), tensor([[100, 137],
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':   # 正样本用的地方  sample = [[68,16,94], [68,16,121],[68,16,147],[68,16,117]]
            batch_size, negative_sample_size = sample.size(0), 1      # 4   1 

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)              # torch.Size([4, 1, 128])

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)              # shape:torch.Size([4, 1, 64])

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)     # shape:torch.Size([4, 1, 128])

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = sample    #  head_part->tensor([[68,16,94], [68,16,121],[68,16,147],[68,16,117]])  tail_part->tensor([[100,137],[31,212],[96,159],[83,188]])
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)  # batch_size=4     negative_sample_size=2 

            head = torch.index_select(       # head (4,1,128)    从48行中抽取头实体对应的词向量  entity_embedding [225,128]  4个相同的头实体对应4个相同的128维词向量
                self.entity_embedding,       # shape:torch.Size([225, 128])
                dim=0,
                index=head_part[:, 0]        # index：（，）  head_part[:, 0]代表这个batch里面所有的头实体的编号，如[68,68,68,68]  
            ).unsqueeze(1)                   # （第0维的大小与index的相同，其他的维度与inout的相同）  (4,128) >> (4,1,128)  代表4个128维的向量 

            relation = torch.index_select(   # 根据关系的编号拿出对应的向量
                self.relation_embedding,     # relation(4,1,64)  从57行中抽取关系对应的词向量  relation_embedding [17,64] 
                 dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)                   # (4,64) >> (4,1,64  代表4个64维的向量

            tail = torch.index_select(
                self.entity_embedding,       # shape:torch.Size([225, 128])
                dim=0,
                index=tail_part.view(-1)     #  tail_part[4×2] index = [8,]  共有4×2=8个索引 把这些尾实体编号对应的向量都拿出来
            ).view(batch_size, negative_sample_size, -1) # (8,128) >> (4,2, 128)   即还原回去tail_part->([[100,137],[31,212],[96,159],[83,188]]) 每个索引变成对应向量

        else:
            raise ValueError('mode %s not supported' % mode)

        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)   #  传入的head relation  是正确的   tail是假的负样本
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score

    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim=2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim=2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2) # 分块  实数域和复数域
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation / (self.embedding_range / pi)    # 一个trick,目的应该是把实体和关系拉齐到同一级别（由于关系这里进行了cos/sin计算）
                                                                   # 关系的复数通过欧拉方程实现   cos(θ) + isin(θ)  这样可以限定关系的模长为1  phase_relation=θ
        re_relation = torch.cos(phase_relation)  # cos(θ)
        im_relation = torch.sin(phase_relation)  # sin(θ)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation  # 头实体实数部分*关系实数部分 - 头实体虚数 * 关系虚数 =  t。r 的实数部分 torch.Size([4, 2, 64])
            im_score = re_head * im_relation + im_head * re_relation  # 头实体实数部分*关系虚数部分 - 头实体虚数 * 关系实数 =  t。r 的虚数部分           ([4, 1, 64])
            re_score = re_score - re_tail                             # t。r - h  实数部分  shape:torch.Size([4, 2, 64])
            im_score = im_score - im_tail                             # t。r - h  虚数部分  shape:torch.Size([4, 2, 64])

        score = torch.stack([re_score, im_score], dim=0)  #shape:torch.Size([2, 4, 2, 64]) 实数复数合并
        score = score.norm(dim=0)                         # shape:torch.Size([4, 2, 64])

        score = self.gamma.item() - score.sum(dim=2)# shape:torch.Size([4, 2]) 向量变成具体数值
        return score

    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846

        # Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head / (self.embedding_range / pi)
        phase_relation = relation / (self.embedding_range / pi)
        phase_tail = tail / (self.embedding_range / pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim=2) * self.modulus
        return score

    @staticmethod# # [(68, 16, 94), (68, 16, 121), (68, 16, 147), (68, 16, 117), (68, 16, 180)......]
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train() # 让你的模型知道现在正在训练。像dropout、batchnorm 层在训练和测试时的作用不同，所以需要使它们运行在对应的模式中。

        optimizer.zero_grad()# 每一轮batch需要设置optimizer.zero_grad，根据pytorch中的backward()函数的计算，当网络参量进行反馈时，梯度是被积累的而不是被替换掉；但是在每一个batch时毫无疑问并不需要将两个batch的梯度混合起来累积，因此这里就需要每个batch设置一遍zero_grad 了。
                             # 
        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)  # 第1轮 'tail-batch'   tensor([[68,16,94], [68,16,121],[68,16,147],[68,16,117]])  tensor([[100,137],[31,212],[96,159],[83,188]])
 # 取出正负样本       positive_sample:tensor:(4,3) 这是batch个正确的三元组       negative_sample:tensor:(4,2) 这是针对这batch个正确三元组的2个错误的尾/头实体
        if args.cuda:
            positive_sample = positive_sample.cuda()   # 正确的三元组
            negative_sample = negative_sample.cuda()   # 对于每个正确的三元组的一组错误的尾实体
            subsampling_weight = subsampling_weight.cuda() # 在采样过程中以一定的概率丢弃一些？
           # (positive_sample, negative_sample) 组成sample
        negative_score = model((positive_sample, negative_sample), mode=mode) # 'tailbatch'(4,2) 2个负样本的尾实体，针对每一个尾实体有一个负样本的分数  最后返回[4,2]的数值->分数

        if args.negative_adversarial_sampling: # 是否加自对抗的负采样策略，也就是对应：F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach()
            # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                              * F.logsigmoid(-negative_score)).sum(dim=1)                  
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)  # [4]  四个值 代表负样本对应实体处的损失          # 公式中的(4)有部分  还没加负号

        positive_score = model(positive_sample)

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)    # [4]  四个值 目前的头 关系 尾 向量计算得分       # 公式中的(4)左部分  也没加负号

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()   # 公式中的(4)中的负号在这里加的
            negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        if args.regularization != 0.0:
            # Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                    model.entity_embedding.norm(p=3) ** 3 +
                    model.relation_embedding.norm(p=3).norm(p=3) ** 3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log

    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''

        model.eval()

        if args.countries:
            # Countries S* datasets are evaluated on AUC-PR
            # Process test data for AUC-PR evaluation
            sample = list()
            y_true = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            # average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}

        else:
            # Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            # Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples,
                    all_true_triples,
                    args.nentity,
                    args.nrelation,
                    'head-batch'
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num // 2),
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples,
                    all_true_triples,
                    args.nentity,
                    args.nrelation,
                    'tail-batch'
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num // 2),
                collate_fn=TestDataset.collate_fn
            )

            test_dataset_list = [test_dataloader_head, test_dataloader_tail]

            logs = []

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        batch_size = positive_sample.size(0)

                        score = model((positive_sample, negative_sample), mode)
                        score += filter_bias

                        # Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim=1, descending=True)

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            # Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

                            # ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            logs.append({
                                'MRR': 1.0 / ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

        return metrics
