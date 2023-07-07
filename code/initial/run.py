import numpy as np
import pandas as pd
import torch
import argparse
from torch.utils.data import DataLoader
import pickle
import random
import pdb
from tqdm import tqdm

from model import KGEModel
from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator
from utils import Triples


class Runner:
    def __init__(self, params):
        self.params = params  # 提前设定的参数
        data = Triples()      # 得到triples.txt的三元组形式及各种变化   data
        self.entity2id, self.relation2id = data.entity2id, data.relation2id  # entity2id 实体的id(共225个实体 （元素加属性值）)    relation2id( 关系id  17个关系 即属性)
        self.train_triples = data.triples                                    # data中的三元组 (h  r  t) (元素 属性  属性值)

        self.id2entity = {idx: ent for ent, idx in self.entity2id.items()}   # id2entity  id:实体(头,尾) 共225个 (元素108加属性值117)
        self.id2relation = {idx: ent for ent, idx in self.relation2id.items()}  # id:关系(属性) 共17个
        self.params.nentity = len(self.entity2id)                               # 实体数量 225
        self.params.nrelation = len(self.relation2id)                           # 关系数量 17
        print(f'{self.params.nentity} entities, {self.params.nrelation} relations')

        self.kge_model = KGEModel(              # 进入model.py进行初始化  现在这个model可以使用
            model_name=self.params.model,       # 'RotatE'
            nentity=self.params.nentity,        # 225
            nrelation=self.params.nrelation,    # 17
            hidden_dim=self.params.hidden_dim,  # 64
            gamma=self.params.gamma,            # 19.9
            double_entity_embedding=self.params.double_entity_embedding,  # true  即要把实体向量扩大二倍 128
            double_relation_embedding=self.params.double_relation_embedding, # false  关系向量不变 64
        )
        # pdb.set_trace()
        if self.params.cuda:
            self.kge_model = self.kge_model.cuda()  # 让model使用GPU

        self.optimizer = torch.optim.Adam(          # 设定一个阿达玛优化器
            filter(lambda p: p.requires_grad, self.kge_model.parameters()),# filter(函数，序列)函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表
            lr=self.params.learning_rate
        )

        self.train_iterator = self.get_train_iter()
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    def run(self): # 训练一个知识图谱嵌入模型      # 对接run.py 文件最下方Runner.run()函数              &  
        best_result_dict = dict()  
        sum = 0                                                                   # 定义了一个空字典，其变量名为best_result_dict。它可以被用来存储键值对
        for step in range(self.params.max_steps): # max_steps 1000                                  #
            training_logs = []                                                                       #
            log = self.kge_model.train_step(self.kge_model,                                          #
                                            self.optimizer,                                          #
                                            self.train_iterator,                                     #
                                            self.params)                                              #
            # training_logs.append(log)                                                               #
            # print(log)                                                                              #  
                                                                                        #  
            print(f"[{step}] Loss={log['loss']:.5f}")
            if(step > 25000):
                sum+= log['loss']
        print(sum)        
        print(sum/5000)                                               # 
        self.save() # 函数调用"self.save()"方法来保存训练好的模型。                                      &
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    def get_train_iter(self):          # 设置训练集头实体的dataloader        头实体，目前这里头和尾定义是完全一样的
        train_dataloader_head = DataLoader(                                # @@@ 在这进入dataloader.py完成TrainDataset类初始化  作为一项参数赋值给train_dataloader_head  
            TrainDataset(self.train_triples, self.params.nentity, self.params.nrelation,  # 22行定义的triples 即(元素，属性， 属性值)  实体数量225    关系数量17
                         self.params.negative_sample_size, 'head-batch'),                 # 负样本数量为2  '使用 head-batch'
            batch_size=self.params.batch_size,                                            # batchsize  4                                    
            shuffle=False,
            num_workers=max(1, self.params.cpu_num // 2),
            collate_fn=TrainDataset.collate_fn # collate_fn：dataloader.py 70行  表示合并样本列表以形成小批量的Tensor对象,如何取样本的，我们可以定义自己的函数来准确地实现想要的功能
        )

        train_dataloader_tail = DataLoader(
            TrainDataset(self.train_triples, self.params.nentity, self.params.nrelation,
                         self.params.negative_sample_size, 'tail-batch'),
            batch_size=self.params.batch_size,
            shuffle=False,
            num_workers=max(1, self.params.cpu_num // 2),
            collate_fn=TrainDataset.collate_fn
        )
        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)  # 将两个PyTorch数据加载器（dataloader）转换为Python迭代器，并以交替方式从它们中获取数据
        return train_iterator      # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%这个方法就是用来抽取训练用的数据%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def save(self):            # 将最终结果存入RotatE——128——64文件中
        with open(f'{self.params.model}_{self.kge_model.entity_dim}_{self.kge_model.relation_dim}.pkl', 'wb') as f:
            dict_save = {
                'id2entity': self.id2entity,
                'id2relation': self.id2relation,
                'entity': self.kge_model.entity_embedding.data,
                'relation': self.kge_model.relation_embedding.data
            }
            pickle.dump(dict_save, f)


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    parser.add_argument('--random_seed', default=1234, type=int)
    parser.add_argument('--model', default='RotatE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true', default=True)
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')

    parser.add_argument('-n', '--negative_sample_size', default=2, type=int)
    parser.add_argument('-d', '--hidden_dim', default=64, type=int)
    parser.add_argument('-g', '--gamma', default=19.9, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=4, type=int)
    parser.add_argument('-r', '--regularization', default=1e-9, type=float)
    parser.add_argument('--uni_weight', action='store_true',
                        help='Otherwise use subsampling weighting like in word2vec')

    parser.add_argument('-lr', '--learning_rate', default=0.025, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('--max_steps', default=30000, type=int)
    parser.add_argument('--log_steps', default=2, type=int, help='train log every xx steps')

    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')

    return parser.parse_args(args)


def main():
    params = parse_args()  # 
    result_dict_list = []

    # torch.manual_seed(params.random_seed)
    # np.random.seed(params.random_seed)
    # torch.cuda.manual_seed(params.random_seed)
    # random.seed(params.random_seed)
    runner = Runner(params)
    runner.run()


'''
CUDA_VISIBLE_DEVICES=1 python run.py -adv -de
'''
main()
#
# params = parse_args()
# params.fold=0
# runner = Runner(params)
# runner.evaluate()
