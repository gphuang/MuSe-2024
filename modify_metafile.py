import pandas as pd
import sys

# modify meta file to use devel data for training

# devel meta ids
from data_parser import get_data_partition

task='c1_muse_perception' # 'c2_muse_humor' # 
meta_fname=f'/scratch/elec/puhe/c/muse_2024/{task}/metadata/partition.csv'

_, partition_to_subject = get_data_partition(meta_fname)
print(len(partition_to_subject['train']), len(partition_to_subject['devel']), len(partition_to_subject['test']))

df=pd.read_csv(meta_fname)
print(df.shape, df.head(3))

df1 = df[df['Partition']=='devel'].copy()
#print(df1.shape)
df1.Partition=df1.Partition.replace('devel', 'train')
df2=pd.concat([df1, df], ignore_index=True, sort=False)
print(df2.shape, df2[df2.Partition=='train'].shape, df2.head(3))
df2.to_csv(f'/scratch/work/huangg5/muse/MuSe-2024/results/csvs/{task}-traindevel.csv')