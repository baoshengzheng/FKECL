#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.utils.data import Dataset

                              # init初始化发生在run.py 63行  73 行 
class TrainDataset(Dataset): #[(68, 16, 94), (68, 16, 121), (68, 16, 147), (68, 16, 117), (68, 16, 180), 
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode): # 22行定义的triples 即(元素，属性， 属性值) 实体数量225    
        self.len = len(triples)   #  head-batch 1643                                              # 关系数量17  负样本数量为2  '使用 head-batch'
        self.triples = triples    #   
        self.triple_set = set(triples)  #   
        self.nentity = nentity                  # 225
        self.nrelation = nrelation              # 17
        self.negative_sample_size = negative_sample_size  # 2
        self.mode = mode              # 是head-batch'  还是tail-batch'
        self.count = self.count_frequency(triples) # 获取部分三元类（head，relation）或（relation，tail）的频率  @函数在78行  @ 共统计的1743行三元组：count 代表出现过的频率{(68, 16): 10, (94, -17): 4, (121, -17): 4, (147, -17): 4
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)# 统计(头实体，关系)的尾实体有什么以及(关系，尾实体)的头实体有什么  @函数在97行

    def __len__(self):  # 返回样本个数
        return self.len

    def __getitem__(self, idx):
        positive_sample = self.triples[idx]

        head, relation, tail = positive_sample

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation - 1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size * 2)
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_head[(relation, tail)],
                    assume_unique=True,
                    invert=True
                )
            elif self.mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_tail[(head, relation)],
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]

        negative_sample = torch.LongTensor(negative_sample)

        positive_sample = torch.LongTensor(positive_sample)

        return positive_sample, negative_sample, subsampling_weight, self.mode

    @staticmethod
    def collate_fn(data):  # run.py 70行
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode

    @staticmethod
    def count_frequency(triples, start=4): # [(68, 16, 94), (68, 16, 121), (68, 16, 147)....]   # 对应self.count    为什么start是4？  
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)  # 统计用于下采样的频率
        The frequency will be used for subsampling like word2vec  
        '''
        count = {}
        for head, relation, tail in triples:  # 遍历整个triples 第一轮(68, 16, 94) 依次赋值给head rela tail   第2轮 (68, 16, 121)
            if (head, relation) not in count:
                count[(head, relation)] = start     # 1轮{(68, 16): 4}  
            else:
                count[(head, relation)] += 1                                           # 2轮{(68, 16): 5, (94, -17): 4}                第3轮 {(68, 16): 6, (94, -17): 4, (121, -17): 4}

            if (tail, -relation - 1) not in count:
                count[(tail, -relation - 1)] = start # 1轮{(68, 16): 4, (94, -17): 4}  2轮{(68, 16): 5, (94, -17): 4, (121, -17): 4}  第3轮{(68, 16): 6, (94, -17): 4, (121, -17): 4, (147, -17): 4}
            else:
                count[(tail, -relation - 1)] += 1
        return count

    @staticmethod
    def get_true_head_and_tail(triples):  # 统计(头实体，关系)的尾实体有什么以及(关系，尾实体)的头实体有什么，返回字典
        '''
        Build a dictionary of true triples that will    建立一个真三元组字典
        be used to filter these true triples for negative sampling   用于过滤负采样的真三元组
        '''

        true_head = {}
        true_tail = {}

        for head, relation, tail in triples: # [(68, 16, 94), (68, 16, 121), (68, 16, 147), (68, 16, 117), (68, 16, 180)......]
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []                            # 第1轮 {(68, 16): []}
            true_tail[(head, relation)].append(tail)  # {头，关系: [尾的列表]} 第1轮 {(68, 16): [94]}   第2轮 {(68, 16): [94, 121]}             第3轮 {(68, 16): [94, 121, 147]}
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []                            # 第1轮 {(16, 94): []}     第2轮 {(16, 94): [68], (16, 121): []}   第3轮 {(16, 94): [68], (16, 121): [68], (16, 147): []}
            true_head[(relation, tail)].append(head)  # {关系，尾: [头的列表]} 第1轮 {(16, 94): [68]}   第2轮 {(16, 94): [68], (16, 121): [68]}  第3轮 {(16, 94): [68], (16, 121): [68], (16, 147): [68]}
# 把list转成ndarray的格式 {(68, 16): [94, 121, 147]}  >>>  {(68, 16): array([94, 121, 147])}
        for relation, tail in true_head:  # 后面用到np.in1d，因此要转化成ndarray
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)]))) # 去重并且用np.array(list)方法来创建ndarray数组
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

        return true_head, true_tail


class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        if self.mode == 'head-batch':
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.nentity)]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)

        tmp = torch.LongTensor(tmp)
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, relation, tail))

        return positive_sample, negative_sample, filter_bias, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode


class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head) # 它将两个PyTorch数据加载器（dataloader）转换为Python迭代器，并以交替方式从它们中获取数据
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0
# 获取数据。其主要方法是__next__，它返回一个数据批次并增加一个步骤计数器。当步骤计数器为偶数时，该方法从迭代器iterator_head中获取下一个数据批次，否则从迭代器iterator_tail中获取下一个数据批次。如果迭代器达到末尾，则再次从头开始循环迭代器。
    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader: # 定义了一个静态方法one_shot_iterator，该方法将PyTorch数据加载器转换为Python迭代器。
                yield data          # 它通过在数据加载器上迭代，使用yield语句将每个数据批次返回为迭代器的元素。由于该方法具有无限循环的特性，因此需要在迭代器中手动停止迭代，否则将一直循环下去。