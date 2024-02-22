# FKECL
Release Date: 7th july, 2023

Author
```
~Baosheng Zheng (Northeastern University in CHINA)
~Changyong Yu (Northeastern University in CHINA)
```

## 1.Introduction
Molecular representation learning and property prediction is a key step in drug discovery and design. Graph contrastive learning-based approaches have gained popularity as a viable solution strategy in recent publications. Most of these methods work with the molecular graph in which the nodes are atoms, and the edges represent the chemical bonds. However, the functional groups are more relative to the properties of the molecules. Moreover, the general-purpose graph augmentation method becomes inefficient due to the meaninglessness of graphs. To address these problems, we proposed a new Functional Group Knowledge-Enhanced Contrastive Learning(FKECL) framework for molecular representation learning. It consists of four modules: 1) We propose a functional group-based graph model in which functional groups are nodes; 2) We construct a functional groups knowledge graph (FGKG) which describes the relationships between functional groups and their basic chemical properties. Under the guidance of FGKG, we expand the functional group graph to an augmentation graph; 3) We extract molecular representations with a common graph encoder for the functional group graph; and a Knowledge Graph Neural Network (KGNN) encoder is used to encode the complex information in the augmented molecular graph; 4) We construct the contrastive objective in which we maximize agreement between these two views of molecular graphs. Experimental results showed that FKECL performed better than state-of-the-art baselines for downstream tasks on six molecular datasets.

## 2.Test Data
### Pre-training data
We use 250K molecules from the ZINC 15 datasets to pre-train FKECL。The pre-training data can be found in `data/raw/zinc15_250K_2D.csv`. As for downstream performance evaluation, we use datasets from MoleculeNet , you can find in `data/raw/bbbp.csv`，`data/raw/sider.csv`，`data/raw/tox21.csv`，`data/raw/clintox.csv`.

Please execute `cd data` and run:
```
 -  `python graph_utils.py`
```
Then you can find the processed LMDB file in `zinc15_250K_2D`. And you can also find `zinc15_250K_2D.pkl`, which determines the order in which the pre-training molecules are read.

### Knowledge feature initialization

We adopt RotateE to train Chemical Element KG (`code/triples.txt`), the resulting embedding file is stored in `code/initial/RotatE_128_64_emb.pkl`.
If you want to train the KG by yourself, please execute `cd code/initial` and run:
```
-   `python load.py`.
```

## 3.Running
After pre-training,  you can test on downstream tasks, please execute `cd code` and run:
```
-   `bash script/finetune.sh`
```
Change the `data_name` command in the bash file to replace different datasets.
You can also specify the `encoder_name`, `training rate`, `encoder path`, `readout_path`, etc. in this bash file.
```
such as: 
CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --seed 12 \
    --encoder_name GNN \
    --batch_size 64 \
    --predictor_hidden_feats 32 \
    --patience 100 \
    --encoder_path ./dump/Pretrain/gnn-kmpnn-model/GCNNodeEncoder_0910_0900_2000th_epoch.pkl \
    --readout_path ./dump/Pretrain/gnn-kmpnn-model/WeightedSumAndMax_0910_0900_2000th_epoch.pkl \
    --lr 0.001 \
    --predictor nonlinear \
    --eval nonfreeze \
    --data_name Tox21 \
    --dump_path ./dump \
    --exp_name KG-finetune-gnn \
    --exp_id tox21
```
## 4.Parameter Settings
We implement FKECL in Pytorch, a publicly available deep learning tool. In the pre-training stage, the Adam optimizer is applied to train FKECL while the initial learning rate is set to 1e-4, and the batch size is set to 256. The number of pre-training epochs is fixed at 80 epochs. For downstream tasks an additional MLP or a linear classifier to predict the molecule property was appended respectively. We froze the weights of the FKECL encoders and tuned MLP or linear classifier in the fine-tuning stage. We still use the Adam optimizer; the learning rate is set to 0.01, and the batch size is set to 64. Furthermore, we use early stopping on the validation set.

## 5.Contacts
Please e-mail your feedback at 2272229@stu.neu.edu.cn.
