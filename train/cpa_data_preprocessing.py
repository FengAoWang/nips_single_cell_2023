import anndata as ad
import pandas as pd
import scanpy as sc
import random

# 将数据转换为adata

''' 
    adata.obs['split']: split_key
       split the dataset into three subsets: 
       train, test, and ood(Out-Of-Distribution)
'''

de_train = pd.read_parquet('/opt/data/private/nips_single_cell_2023-master/data/de_train.parquet')
id_map = pd.read_csv('/opt/data/private/nips_single_cell_2023-master/data/id_map.csv')
sample_submission = pd.read_csv('/opt/data/private/nips_single_cell_2023-master/data/sample_submission.csv', index_col='id')

adata = sc.AnnData(X=de_train.iloc[:, 5:].values)  

adata.obs['cell_type'] = de_train['cell_type'].tolist()
adata.obs['sm_name'] = de_train['sm_name'].tolist()

# 生成 'train' 和 'valid' 的列表
total_samples = adata.shape[0]
train_count = int(total_samples * 0.9)
valid_count = total_samples - train_count
split_data = ['train'] * train_count + ['valid'] * valid_count

# 打乱列表中的元素顺序
random_seed = 46
random.seed(random_seed)
random.shuffle(split_data)

adata.obs['split'] = split_data

# test data
adata_test = sc.AnnData(X=sample_submission.values) 
adata_test.obs['cell_type'] = id_map['cell_type'].tolist()
adata_test.obs['sm_name'] = id_map['sm_name'].tolist()

adata_test.obs['split'] = ['ood'] * adata_test.shape[0]

adata_combined = ad.concat([adata, adata_test])
adata_combined.var_names = de_train.columns[5:]
adata_combined.var['gene'] = de_train.columns[5:].tolist()

adata_combined.write_h5ad('/opt/data/private/nips_single_cell_2023-master/data/processed_data/adata.h5ad')

print('end')