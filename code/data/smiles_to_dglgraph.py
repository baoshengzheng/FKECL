from rdkit import RDLogger
import logging
logger = logging.getLogger()
RDLogger.DisableLog('rdApp.*')
from rdkit import Chem
import dgl
import torch
import pdb
from .load_triples import Triples


def bondtype_features(bond):
    bondtype_list=['SINGLE','DOUBLE','TRIPLE','AROMATIC']
    bond2emb = {}
    for idx, bt in enumerate(bondtype_list):
        bond2emb[bt]= idx
    fbond = bond2emb[str(bond.GetBondType())]
    return fbond

def smiles_2_kgdgl(smiles):
    data = Triples()
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        print('因为宝圣踢球太好所以无法打印')  # 如果无法解析，则打印一条错误信息并返回None。
        return None
    
    
    # 构建 SMARTS 字符串 
    smarts = '*-[N;R0]=[C;D1;H2] & *-[S;D4](=O)(=O)-[O;D2]-[C;D1;H3] & *-[N;D2]=[C;D2]=[O;D1] & *-C(=O)-[N;D1] & *-[N;D2]-[S;D4](=[O;D1])(=[O;D1])-[C;D1;H3] & *-[S;D4](=O)(=O)-[O;D1] & *=[O;D1] & *=[S;D1] & [#0]-[#17] & *-[S;D2]-[C;D1;H3] & *-[O;D2]-[C;D2]-[C;D1;H3] & *-[C;D2]#[N;D1] & *-[N;D2]-[C;D3](=O)-[C;D1;H3] & *-[S;D1] & *-[C;D2]#[C;D1;H] & *-[C;D4](F)(F)F & [#0]-[#53] & *=[O;D1] & [#0]-[#35] & *=[N;D1] & *=[N;R0]-[C;D1;H3] & *-[N;D2]=[N;D2]-[C;D1;H3] & [#0]-[#9] & *-[N;D2]=[C;D2]=[S;D1] & *-[O;D1] & *-C(=O)[O;D2]-[C;D1;H3] & *#[N;D1] & *-C(=O)-[C;D1;H3] & *-C(=O)[O;D1] & *-[N;D2]=[N;D1] & *-[S;D4](=[O;D1])(=[O;D1])-[N;D1] & *-[S;D4](=O)(=O)-[C;D1;H3] & *-C(=O)-[C;D1] & *-[C;D4]([C;D1])([C;D1])-[C;D1] & *=[N;R0]-[O;D1] & *-[O;D2]-[C;D1;H3] & *-[S;D4](=O)(=O)-[Cl] & *-[C;D3]1-[C;D2]-[C;D2]1'

    # 编译 SMARTS 字符串为模式对象列表
    pattern_list = [Chem.MolFromSmarts(smart) for smart in smarts.split(' & ')]

    # Create a dictionary to store the relationship between each match and its corresponding SMARTS string
    match_dict = {}
    # 搜索模式对象在 RDKit 分子对象中的匹配

    match_atoms = set() # match_atoms中的官能团节点
    all_func_gro_features = {}
    for pattern in pattern_list:
        print('***************')
        matches = mol.GetSubstructMatches(pattern)
        sub_smarts = Chem.MolToSmarts(pattern)
        print(f'{sub_smarts}: {matches}')
        if matches:
            print('含有该结构')

            # 打印每个匹配的原子索引
            for match in matches:
        
                print("目前在遍历",end = '')
                print(match)
                print('match中的第0个原子代表*  暂且不算  想算的话去掉if即可')
                # Print the atom indices for each match
                func_gro_feature = 0
                for i, match_atom_index in enumerate(match):
                    print(f'遍历当前match的第{i}个值: {match_atom_index}')

                    if i == 1:
                        func_gro_index = match_atom_index
                        match_atoms.add(match_atom_index)
                    if i!=0:
                        atom_in_funcgro = mol.GetAtomWithIdx(match_atom_index)
                        atomic_num_in_funcgro = atom_in_funcgro.GetAtomicNum()
                        func_gro_feature += atomic_num_in_funcgro
                        
                        # Convert the substructure molecule to a SMARTS string
                        sub_smiles = Chem.MolToSmiles(pattern)
                        
                        # Add the match and SMARTS string to the dictionary
                        match_dict[match_atom_index] = sub_smiles
                all_func_gro_features[func_gro_index] = func_gro_feature # 该官能团节点的特征 以官能团起始节点做为官能团节点  官能团节点特征用这个官能团内部所有正常节点特征(原子序数)相加

    connected_atom_list = [] # 这里创建了一个名为connected_atom_list的列表，用于存储所有连接的原子索引。
    for bond in mol.GetBonds():# 遍历了分子中的所有化学键，然后将连接的原子的索引添加到connected_atom_list列表中，
        if bond.GetBeginAtomIdx() not in match_atoms and bond.GetEndAtomIdx() not in match_atoms:
            connected_atom_list.append(bond.GetBeginAtomIdx())               # 对于每一个键，将键的开始原子和结束原子的索引分别加入到connected_atom_list中。
            connected_atom_list.append(bond.GetEndAtomIdx())
        if bond.GetBeginAtomIdx() in match_atoms and bond.GetEndAtomIdx() not in match_atoms:
            connected_atom_list.append(bond.GetBeginAtomIdx())               # 对于每一个键，若键的开始原子是官能团节点和结束原子是正常节点 索引分别加入到connected_atom_list中。
            connected_atom_list.append(bond.GetEndAtomIdx())
        if bond.GetBeginAtomIdx() not in match_atoms and bond.GetEndAtomIdx() in match_atoms:
            connected_atom_list.append(bond.GetBeginAtomIdx())               # 对于每一个键，将键的开始原子是正常节点和结束原子是官能团节点的索引分别加入到connected_atom_list中。
            connected_atom_list.append(bond.GetEndAtomIdx())       
    
    connected_atom_list = sorted(list(set(connected_atom_list)))  # 用于存储所有连接的原子索引然后将这个列表去重排序

    connected_atom_map = {k: v for k, v in zip(connected_atom_list, list(range(len(connected_atom_list))))}#创建一个名为connected_atom_map的字典，用于将原子索引映射到一个数字编码。
    atoms_feature = [0 for _ in range(len(connected_atom_list))]  # 创建一个名为atoms_feature的列表，[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...]用于存储每个原子(包含官能团节点)的特征向量。 

    # get all node ids and relations  连入知识图谱
    begin_atoms = []# 存储头实体的节点索引
    end_entities = []# 存储尾实体的标识符
    rel_features = []# 存储关系的特征
    for atom in mol.GetAtoms(): # 遍历SMILES分子中的每个原子'C'
        node_index = atom.GetIdx()  # 获取原子在分子中的索引 0
        if node_index not in connected_atom_list: # 如果原子不在连接原子列表中，则跳过该原子
            continue

        # 当前节点遍历到普通节点
        if node_index not in match_atoms:

            atomicnum = atom.GetAtomicNum() # 获取原子的原子序数 6
            atoms_feature[connected_atom_map[node_index]] = atomicnum  # 更新连接节点列表中该正常原子对应的特征
        
        # 当前节点遍历到官能团节点
        if node_index in match_atoms and node_index in connected_atom_list:  # and node_index in connected_atom_list 可以不加 方便理解加上了

            atoms_feature[connected_atom_map[node_index]] = all_func_gro_features[node_index] # 更新连接节点列表中官能团节点对应的特征

            symbol = match_dict[node_index]# 获取官能团节点所代表的官能团符号  
            
    # 如果该原子符号在三元组中
            if symbol in data.entities:
                tid = [t for (r,t) in data.h2rt[data.entity2id[symbol]]] # 获取该实体对应的所有关系的标识符和特征
                rid = [r for (r,t) in data.h2rt[data.entity2id[symbol]]]
    # 将头实体的节点索引、尾实体的标识符和关系的特征分别添加到对应的列表中
                begin_atoms.extend([node_index]*len(tid))   # add head entities
                end_entities.extend(tid)    # add tail eneities   这个官能团节点对应的尾实体
                rel_features.extend(i+4 for i in rid)#  pytorch-geometric 中预定义的关系编号是从 0 到 3 分别代表着"被连接"、"连接"、"连接后的共享电子对"和"环"等四种类型的关系

    # get list of tail entity ids and features
    if end_entities:
        entity_id = sorted(list(set(end_entities)))
        node_id = [i+len(connected_atom_list) for i in range(len(entity_id))]  # 尾实体id  加18 是因为前0-17 为头节点id   这个connected_atom_list 同时包含了分子图中普通分子和官能团节点  在这个基础上加得nodeid 即官能团属性值的id
        entid2nodeid = dict(zip(entity_id, node_id)) # dict: t_id in triples --> node_id in dglgraph    得到（尾实体：尾实体id）相对应的字典
        nodeids = [entid2nodeid[i] for i in end_entities]   # list of tail entity id    根据字典将全部尾实体列表转化对应id的列表  加完长度之后的

        nodes_feature = [i+118 for i in entity_id]   # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& 104 要改

    # get list of atom ids and bond features
    begin_indexes = []
    end_indexes = []
    bonds_feature = []
    edge_type = []
    
    for bond in mol.GetBonds():
        bond_feature = bondtype_features(bond)
        if bond.GetBeginAtomIdx() in match_atoms and bond.GetEndAtomIdx() in match_atoms: 
            continue

        begin_indexes.append(connected_atom_map[bond.GetBeginAtomIdx()])
        end_indexes.append(connected_atom_map[bond.GetEndAtomIdx()])
        bonds_feature.append(bond_feature)

        begin_indexes.append(connected_atom_map[bond.GetEndAtomIdx()])
        end_indexes.append(connected_atom_map[bond.GetBeginAtomIdx()])
        bonds_feature.append(bond_feature)
        
        # if connected_atom_map[bond.GetBeginAtomIdx()] not in match_atoms and connected_atom_map[bond.GetEndAtomIdx()] not in match_atoms: 
        #     begin_indexes.append(connected_atom_map[bond.GetBeginAtomIdx()])
        #     end_indexes.append(connected_atom_map[bond.GetEndAtomIdx()])
        #     bonds_feature.append(bond_feature)

        #     begin_indexes.append(connected_atom_map[bond.GetEndAtomIdx()])
        #     end_indexes.append(connected_atom_map[bond.GetBeginAtomIdx()])
        #     bonds_feature.append(bond_feature)
        
        # if connected_atom_map[bond.GetBeginAtomIdx()] not in match_atoms and connected_atom_map[bond.GetEndAtomIdx()] in match_atoms:
        #     begin_indexes.append(connected_atom_map[bond.GetBeginAtomIdx()])
        #     end_indexes.append(connected_atom_map[bond.GetEndAtomIdx()])
        #     bonds_feature.append(bond_feature)        
     
        #     # begin_indexes.append(connected_atom_map[bond.GetEndAtomIdx()])
        #     # end_indexes.append(connected_atom_map[bond.GetBeginAtomIdx()])
        #     # bonds_feature.append(bond_feature)
        
        # if connected_atom_map[bond.GetBeginAtomIdx()] in match_atoms and connected_atom_map[bond.GetEndAtomIdx()] not in match_atoms:
        #     begin_indexes.append(connected_atom_map[bond.GetBeginAtomIdx()])
        #     end_indexes.append(connected_atom_map[bond.GetEndAtomIdx()])
        #     bonds_feature.append(bond_feature)        
     
        #     # begin_indexes.append(connected_atom_map[bond.GetEndAtomIdx()])
        #     # end_indexes.append(connected_atom_map[bond.GetBeginAtomIdx()])
        #     # bonds_feature.append(bond_feature) 
    edge_type.extend([0]*len(bonds_feature))     # 总边数

    # add ids and features of tail entities and relations
    if begin_atoms:
        begin_indexes.extend(nodeids) # change head and tail entities
        end_indexes.extend(begin_atoms)
        atoms_feature.extend(nodes_feature)
        bonds_feature.extend(rel_features)
        edge_type.extend([1]*len(rel_features))

    # create dglgraph
    graph = dgl.graph((begin_indexes, end_indexes), idtype=torch.int32)  # 图中节点   idtype 这个位置好像可以根据GPU确定单精还是半精？ 再议再议
    graph.edata['e'] = torch.tensor(bonds_feature, dtype=torch.long)  # 图中边
    graph.ndata['h'] = torch.tensor(atoms_feature, dtype=torch.long)  # 节点性质
    graph.edata['etype'] = torch.tensor(edge_type, dtype=torch.long) # 0 for bonds & 1 for rels  # 图中边的性质
    print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
    return graph

       
def smiles_2_dgl(smiles):   
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print('Invalid mol found/ 222222')
        return None
        
    connected_atom_list = []
    for bond in mol.GetBonds():
        connected_atom_list.append(bond.GetBeginAtomIdx())
        connected_atom_list.append(bond.GetEndAtomIdx())
        
    connected_atom_list = sorted(list(set(connected_atom_list)))    
    connected_atom_map = {k: v for k, v in zip(connected_atom_list, list(range(len(connected_atom_list))))}
    atoms_feature = [0 for _ in range(len(connected_atom_list))]
    
    # get all node ids and relations
    for atom in mol.GetAtoms():
        node_index = atom.GetIdx() # 返回原子在其所属分子中的唯一索引
        atomicnum = atom.GetAtomicNum()
        if node_index not in connected_atom_list:
            continue
        
        atoms_feature[connected_atom_map[node_index]] = atomicnum    
             
    # get list of atom ids and bond features
    begin_indexes = []
    end_indexes = []
    bonds_feature = []
    
    for bond in mol.GetBonds():
        bond_feature = bondtype_features(bond)
        
        begin_indexes.append(connected_atom_map[bond.GetBeginAtomIdx()])
        end_indexes.append(connected_atom_map[bond.GetEndAtomIdx()])
        bonds_feature.append(bond_feature)

        begin_indexes.append(connected_atom_map[bond.GetEndAtomIdx()])
        end_indexes.append(connected_atom_map[bond.GetBeginAtomIdx()])
        bonds_feature.append(bond_feature)
    
    # create dglgraph
    graph = dgl.graph((begin_indexes, end_indexes), idtype=torch.int32)
    graph.edata['e'] = torch.tensor(bonds_feature, dtype=torch.long)
    graph.ndata['h'] = torch.tensor(atoms_feature, dtype=torch.long)
    
    return graph

# def smiles_2_dgl(smiles):
#     mol = Chem.MolFromSmiles(smiles)

#     if mol is None:
#         print('因为宝圣踢球太好所以无法打印')  # 如果无法解析，则打印一条错误信息并返回None。
#         return None
    
    
#     # 构建 SMARTS 字符串 
#     smarts = '*-[N;R0]=[C;D1;H2] & *-[S;D4](=O)(=O)-[O;D2]-[C;D1;H3] & *-[N;D2]=[C;D2]=[O;D1] & *-C(=O)-[N;D1] & *-[N;D2]-[S;D4](=[O;D1])(=[O;D1])-[C;D1;H3] & *-[S;D4](=O)(=O)-[O;D1] & *=[O;D1] & *=[S;D1] & [#0]-[#17] & *-[S;D2]-[C;D1;H3] & *-[O;D2]-[C;D2]-[C;D1;H3] & *-[C;D2]#[N;D1] & *-[N;D2]-[C;D3](=O)-[C;D1;H3] & *-[S;D1] & *-[C;D2]#[C;D1;H] & *-[C;D4](F)(F)F & [#0]-[#53] & *=[O;D1] & [#0]-[#35] & *=[N;D1] & *=[N;R0]-[C;D1;H3] & *-[N;D2]=[N;D2]-[C;D1;H3] & [#0]-[#9] & *-[N;D2]=[C;D2]=[S;D1] & *-[O;D1] & *-C(=O)[O;D2]-[C;D1;H3] & *#[N;D1] & *-C(=O)-[C;D1;H3] & *-C(=O)[O;D1] & *-[N;D2]=[N;D1] & *-[S;D4](=[O;D1])(=[O;D1])-[N;D1] & *-[S;D4](=O)(=O)-[C;D1;H3] & *-C(=O)-[C;D1] & *-[C;D4]([C;D1])([C;D1])-[C;D1] & *=[N;R0]-[O;D1] & *-[O;D2]-[C;D1;H3] & *-[S;D4](=O)(=O)-[Cl] & *-[C;D3]1-[C;D2]-[C;D2]1'

#     # 编译 SMARTS 字符串为模式对象列表
#     pattern_list = [Chem.MolFromSmarts(smart) for smart in smarts.split(' & ')]

#     # Create a dictionary to store the relationship between each match and its corresponding SMARTS string
#     match_dict = {}
#     # 搜索模式对象在 RDKit 分子对象中的匹配

#     match_atoms = set() # match_atoms中的官能团节点
#     all_func_gro_features = {}
#     for pattern in pattern_list:
#         print('***************')
#         matches = mol.GetSubstructMatches(pattern)
#         sub_smarts = Chem.MolToSmarts(pattern)
#         print(f'{sub_smarts}: {matches}')
#         if matches:
#             print('含有该结构')

#             # 打印每个匹配的原子索引
#             for match in matches:
        
#                 print("目前在遍历",end = '')
#                 print(match)
#                 print('match中的第0个原子代表*  暂且不算  想算的话去掉if即可')
#                 # Print the atom indices for each match
#                 func_gro_feature = 0
#                 for i, match_atom_index in enumerate(match):
#                     print(f'遍历当前match的第{i}个值: {match_atom_index}')

#                     if i == 1:
#                         func_gro_index = match_atom_index
#                     if i!=0:
#                         atom_in_funcgro = mol.GetAtomWithIdx(match_atom_index)
#                         atomic_num_in_funcgro = atom_in_funcgro.GetAtomicNum()
#                         func_gro_feature += atomic_num_in_funcgro
#                         match_atoms.add(match_atom_index)
#                         # Convert the substructure molecule to a SMARTS string
#                         sub_smiles = Chem.MolToSmiles(pattern)
                        
#                         # Add the match and SMARTS string to the dictionary
#                         match_dict[match_atom_index] = sub_smiles
#                 all_func_gro_features[func_gro_index] = func_gro_feature # 该官能团节点的特征 以官能团起始节点做为官能团节点  官能团节点特征用这个官能团内部所有正常节点特征(原子序数)相加

#     connected_atom_list = [] # 这里创建了一个名为connected_atom_list的列表，用于存储所有连接的原子索引。

#     for bond in mol.GetBonds():# 遍历了分子中的所有化学键，然后将连接的原子的索引添加到connected_atom_list列表中，
#         if bond.GetBeginAtomIdx() not in match_atoms and bond.GetEndAtomIdx() not in match_atoms:
#             connected_atom_list.append(bond.GetBeginAtomIdx())               # 对于每一个键，将键的开始原子和结束原子的索引分别加入到connected_atom_list中。
#             connected_atom_list.append(bond.GetEndAtomIdx())
#         if bond.GetBeginAtomIdx() in match_atoms and bond.GetEndAtomIdx() not in match_atoms:
#             connected_atom_list.append(bond.GetBeginAtomIdx())               # 对于每一个键，若键的开始原子是官能团节点和结束原子是正常节点 索引分别加入到connected_atom_list中。
#             connected_atom_list.append(bond.GetEndAtomIdx())
#         if bond.GetBeginAtomIdx() not in match_atoms and bond.GetEndAtomIdx() in match_atoms:
#             connected_atom_list.append(bond.GetBeginAtomIdx())               # 对于每一个键，将键的开始原子是正常节点和结束原子是官能团节点的索引分别加入到connected_atom_list中。
#             connected_atom_list.append(bond.GetEndAtomIdx())       
    
#     connected_atom_list = sorted(list(set(connected_atom_list)))  # 用于存储所有连接的原子索引然后将这个列表去重排序

#     connected_atom_map = {k: v for k, v in zip(connected_atom_list, list(range(len(connected_atom_list))))}#创建一个名为connected_atom_map的字典，用于将原子索引映射到一个数字编码。
#     atoms_feature = [0 for _ in range(len(connected_atom_list))]  # 创建一个名为atoms_feature的列表，[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...] 用于存储每个原子(包含官能团节点)的特征向量。 
#     # get all node ids and relations  连入知识图谱

#     for atom in mol.GetAtoms():#遍历SMILES分子中的每个原子'C'
#         node_index = atom.GetIdx()  # 获取原子在分子中的索引 0
#         if node_index not in connected_atom_list: # 如果原子不在连接原子列表中，则跳过该原子
#             continue

#         # 当前节点遍历到普通节点
#         if node_index not in match_atoms:

#             atomicnum = atom.GetAtomicNum() # 获取原子的原子序数 6
#             atoms_feature[connected_atom_map[node_index]] = atomicnum  # 更新连接节点列表中该正常原子对应的特征
        
#         # 当前节点遍历到官能团节点
#         if node_index in match_atoms and node_index in connected_atom_list:  # and node_index in connected_atom_list 可以不加 方便理解加上了

#             atoms_feature[connected_atom_map[node_index]] = all_func_gro_features[node_index] # 更新连接节点列表中官能团节点对应的特征

#     # get list of atom ids and bond features
#     begin_indexes = []
#     end_indexes = []
#     bonds_feature = []
    
#     for bond in mol.GetBonds():
#         bond_feature = bondtype_features(bond)
        
#         # if connected_atom_map[bond.GetBeginAtomIdx()] not in match_atoms and connected_atom_map[bond.GetEndAtomIdx()] not in match_atoms: 
#         #     begin_indexes.append(connected_atom_map[bond.GetBeginAtomIdx()])
#         #     end_indexes.append(connected_atom_map[bond.GetEndAtomIdx()])
#         #     bonds_feature.append(bond_feature)

#         #     begin_indexes.append(connected_atom_map[bond.GetEndAtomIdx()])
#         #     end_indexes.append(connected_atom_map[bond.GetBeginAtomIdx()])
#         #     bonds_feature.append(bond_feature)
        
#         # if connected_atom_map[bond.GetBeginAtomIdx()] not in match_atoms and connected_atom_map[bond.GetEndAtomIdx()] in match_atoms:
#         #     begin_indexes.append(connected_atom_map[bond.GetBeginAtomIdx()])
#         #     end_indexes.append(connected_atom_map[bond.GetEndAtomIdx()])
#         #     bonds_feature.append(bond_feature)        
     
#         #     # begin_indexes.append(connected_atom_map[bond.GetEndAtomIdx()])
#         #     # end_indexes.append(connected_atom_map[bond.GetBeginAtomIdx()])
#         #     # bonds_feature.append(bond_feature)
        
#         # if connected_atom_map[bond.GetBeginAtomIdx()] in match_atoms and connected_atom_map[bond.GetEndAtomIdx()] not in match_atoms:
#         #     begin_indexes.append(connected_atom_map[bond.GetBeginAtomIdx()])
#         #     end_indexes.append(connected_atom_map[bond.GetEndAtomIdx()])
#         #     bonds_feature.append(bond_feature)        
     
#         #     # begin_indexes.append(connected_atom_map[bond.GetEndAtomIdx()])
#         #     # end_indexes.append(connected_atom_map[bond.GetBeginAtomIdx()])
#         #     # bonds_feature.append(bond_feature)
#         if bond.GetBeginAtomIdx() in match_atoms and bond.GetEndAtomIdx() in match_atoms: 
#             continue

#         begin_indexes.append(connected_atom_map[bond.GetBeginAtomIdx()])
#         end_indexes.append(connected_atom_map[bond.GetEndAtomIdx()])
#         bonds_feature.append(bond_feature)

#         begin_indexes.append(connected_atom_map[bond.GetEndAtomIdx()])
#         end_indexes.append(connected_atom_map[bond.GetBeginAtomIdx()])
#         bonds_feature.append(bond_feature)
                

#     # create dglgraph
#     graph = dgl.graph((begin_indexes, end_indexes), idtype=torch.int32)
#     graph.edata['e'] = torch.tensor(bonds_feature, dtype=torch.long)
#     graph.ndata['h'] = torch.tensor(atoms_feature, dtype=torch.long)
    
#     return graph

# if __name__ == '__main__':
#     output_idx = 1
#     data_path = f'./raw/zinc15_250K_2D.csv'
    
#     data = pd.read_csv(data_path)
#     env = lmdb.open(f'./zinc15_250K_2D', map_size=int(1e12), max_dbs=2,lock=True)
    
#     graphs_db = env.open_db('graph'.encode())    # 使用 open_db() 函数打开两个名为 graph 和 kgraph 的子数据库，并将它们分别赋值给 graphs_db 和 kgraphs_db 变量。encode() 函数将字符串转换为字节数组，因为 LMDB 中的键和值都是字节数组。
#     kgraphs_db = env.open_db('kgraph'.encode())  
    
#     graphs = data['smiles'].apply(smiles_2_dgl).to_list()
#     graphs = list(filter(None, graphs))
    
#     kgraphs = data['smiles'].apply(smiles_2_kgdgl).to_list()
#     kgraphs = list(filter(None,kgraphs))
    
#     with env.begin(write=True) as txn:
#         for idx, graph in enumerate(graphs):
#             txn.put(str(idx).encode(), pickle.dumps(graph), db=graphs_db)
#         for idx, kgraph in enumerate(kgraphs):
#             txn.put(str(idx).encode(), pickle.dumps(kgraph), db=kgraphs_db)
#     env.close()
#     print('1')
#     import random
#     i = list(range(len(graphs)))
#     random.shuffle(i)
    
#     with open('zinc15_250K_2D.pkl','wb') as f:
        
#         pickle.dump(i,f)
#         print('2')  