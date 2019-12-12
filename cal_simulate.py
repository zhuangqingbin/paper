import os,time
import pandas as pd
import numpy as np

RecordDir = 'records1'


csv_path_list = []
record_dir = os.path.join(os.getcwd(),RecordDir)
for i in os.listdir(record_dir):
    if len(i) > 10:
        csv_path_list.append(os.path.join(record_dir,i,'record.csv'))

template_df = pd.read_csv(csv_path_list[0])

def init_dict(length):
    tmp_dict = {}
    for i in range(length):
        tmp_dict[i] = []
    return tmp_dict

tr_f1 = init_dict(len(template_df))
tr_acc = init_dict(len(template_df))
tr_auc = init_dict(len(template_df))
te_f1 = init_dict(len(template_df))
te_acc = init_dict(len(template_df))
te_auc = init_dict(len(template_df))


for csv_path in csv_path_list:
    tmp_df = pd.read_csv(csv_path)
    for i in range(len(tmp_df)):
        tr_f1[i].append(tmp_df['train_f1'].iloc[i])
        tr_acc[i].append(tmp_df['train_acc'].iloc[i])
        tr_auc[i].append(tmp_df['train_auc'].iloc[i])
        te_f1[i].append(tmp_df['test_f1'].iloc[i])
        te_acc[i].append(tmp_df['test_acc'].iloc[i])
        te_auc[i].append(tmp_df['test_auc'].iloc[i])

for i in range(len(template_df)):
    template_df['train_f1'] = [np.mean(i) for i in tr_f1.values()]
    template_df['train_acc'] = [np.mean(i) for i in tr_acc.values()]
    template_df['train_auc'] = [np.mean(i) for i in tr_auc.values()]
    template_df['test_f1'] = [np.mean(i) for i in te_f1.values()]
    template_df['test_acc'] = [np.mean(i) for i in te_acc.values()]
    template_df['test_auc'] = [np.mean(i) for i in te_auc.values()]

template_df.to_csv(f'{record_dir}/record.csv',index=False)
