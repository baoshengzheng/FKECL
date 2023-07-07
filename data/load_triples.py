import torch
import pdb
from collections import defaultdict as ddict

class Triples:
    
    def __init__(self, data_dir="./all_triples.txt"):
        self.data = self.load_data(data_dir)
        self.entities, self.entity2id = self.get_entities(self.data)
        self.relations, self.relation2id = self.get_relations(self.data) # 调用get_relations函数获取数据集中所有关系，并给每个关系一个唯一的ID
        self.triples = self.read_triple(self.data, self.entity2id, self.relation2id) # 调用read_triple函数将数据集中的三元组用实体ID和关系ID表示
        self.h2rt = self.h2rt(self.triples) # 调用h2rt函数将三元组按照头实体ID分组，每个头实体对应其出现的关系和尾实体。
    
    def load_data(self, data_dir):   # 读取数据集文件，返回一个列表，其中每个元素是一个三元组列表。
        with open("%s" % (data_dir), "r") as f: # 打开all_triples.txt
            data = f.read().strip().split("\n")# 文件对象的“read”方法读取文件，使用“strip”方法删除任何前导或尾随空格，并使用换行符“\n”作为分隔符将结果字符串拆分为行列表
            data = [i.split() for i in data]
        return data    
    
    def get_relations(self, data):  # 从三元组列表中获取所有关系，并为每个关系分配一个唯一的ID。
        relations = sorted(list(set([d[1] for d in data])))
        relationid = [i for i in range(len(relations))]
        relation2id = dict(zip(relations, relationid))
        return relations, relation2id
    
    def get_entities(self, data):  # 从三元组列表中获取所有实体，并为每个实体分配一个唯一的IDls
        entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
        entityid = [i for i in range(len(entities))]
        entity2id = dict(zip(entities, entityid))
        return entities, entity2id
    
    def read_triple(self, data, entity2id, relation2id): # 将三元组中的实体和关系用其对应的唯一ID替换，返回一个新的三元组列表。
        '''
        Read triples and map them into ids.
        '''
        triples = []
        for triple in data:
            h = triple[0]
            r = triple[1]
            t = triple[2]
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
        return triples
    
    def h2rt(self, triples):    # dict: head_id  --> list[(rel_id, tail_id)]  
        h2rt = ddict(list)      # ：将三元组按照头实体ID分组，每个头实体对应其出现的关系和尾实体。返回一个字典，键为头实体ID，值为该实体的所有关系和尾实体组成的列表。
        for tri in triples:
            h, r, t = tri
            h2rt[h].append((r,t))
        return h2rt

    
if __name__ == '__main__':
    data = Triples()
    print(data.data, '\n') #[['H', 'metallicity', 'lively_nonmetal'], ['H', 'periodic', '1']...]
    print(data.relation2id, '\n')
    print(data.entity2id, '\n')
    print(data.triples)
    print(data.h2rt[82])       