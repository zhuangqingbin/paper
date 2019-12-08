import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re

#########################
##### 处理违约债券数据
#########################
default_bond_rename = {'代码':'bond_code', '债券余额(亿元)':'amount',
                       '最新债项评级':'new_level', '发生日期':'default_date'}
# (504, 4)
default_bond = pd.read_csv('bond_data/default.csv',encoding="gbk")[list(default_bond_rename)].\
    rename(default_bond_rename,axis=1).drop_duplicates()


#########################
##### 处理正常债券数据
#########################
normal_bond_rename = {'代码':'bond_code', '截止日余额(亿元)':'amount',
                      '截止日评级':'new_level'}
# (27806, 3)
normal_bond = pd.read_csv('bond_data/credit_bond.csv',encoding="gbk")[list(normal_bond_rename)].\
    rename(normal_bond_rename,axis=1).drop_duplicates()
normal_bond = normal_bond[~normal_bond.bond_code.isin(default_bond.bond_code.unique())]

#########################
##### 处理债券发行数据
#########################
default_bond_rename = {'代码':'bond_code', '发行日期':'start','到期日':'end',
                       '发行总额(亿元)':'issue_amount', '利率':'interest',
                       '期限(年)':'term','发行价格':'price', '发行时主体评级':'issue_com_level',
                       '发行时债券评级':'issue_level', '票面品种':'kind',
                       '付息利率品种':'pay_kind','发行人':'issuer', '债券类型':'bond_kind',
                       '行业分类':'industry','新发债利率':'new_interest'}
# (62769, 16)
issue_data = pd.read_csv('bond_data/issue.csv',encoding="gbk")[list(default_bond_rename)].\
    rename(default_bond_rename,axis=1)
# 连接发行数据
default_bond = pd.merge(default_bond,issue_data,on='bond_code',how='left')
normal_bond = pd.merge(normal_bond,issue_data,on='bond_code',how='left')

#########################
##### 处理发行主体基本数据
#########################
com_base_info_rename = {'企业名称':'issuer', '最新评级':'com_new_level',
                        '所属省市':'province', '企业性质':'nature',
                        '是否上市':'listed', '成立日期':'pub_date',''
                        '注册资本(万元)':'capital', '所属行业二级':'industry'}
# (7888, 30)
com_base_info = pd.read_csv('bond_data/com_base_info.csv',encoding="gbk")[list(com_base_info_rename)].\
                rename(com_base_info_rename,axis=1)
com_base_info['pub_date'] = com_base_info['pub_date'].astype(str).apply(lambda x:x[:4])
com_base_info['pub_years'] = com_base_info['pub_date'].apply(
    lambda x: 2019-int(x) if len(x)==4 else np.nan)
com_base_info.drop(['pub_date'],axis=1,inplace=True)
# 连接发债主体基本数据
default_bond = pd.merge(default_bond, com_base_info, on='issuer',how='left')
normal_bond = pd.merge(normal_bond, com_base_info, on='issuer',how='left')

#########################
##### 处理发行主体发债数据
#########################
# (7722, 3)
com_info = pd.read_csv('bond_data/com_info.csv',encoding="gbk").\
                rename({'企业名称':'issuer','发债总额(亿元)':'total_amount',
                        '发行只数':'total_num'},axis=1)
# 连接发债主体的债务数据
default_bond = pd.merge(default_bond, com_info, on='issuer',how='left')
normal_bond = pd.merge(normal_bond, com_info, on='issuer',how='left')

#########################
##### 处理发行主体财务数据
#########################
finance_data = pd.read_csv('bond_data/finance.csv',encoding="gbk",low_memory=False).\
                drop(['是否经过审计','审计意见'],axis = 1).\
                rename({'名称':'issuer','报告期':'issue_date'},axis=1).\
                sort_values(['issuer','issue_date']).\
                drop(['EBITDA(亿元)','EBITDA/营业总收入','净资产回报率(%)',
                      '带息债务/总投入资本','经营性现金流/EBITDA',
                      'EBITDA/带息债务'],axis=1)

# 对于正常债券以最新财报为准提取财务数据
normal_bond = pd.merge(normal_bond, finance_data[finance_data.issue_date=='2019-09-30'],
                       on='issuer',how='left').\
                        drop(['issue_date'],axis=1)


# 对于违约债券以违约日期的上一期为准提取财务数据
default_finance_data = finance_data[finance_data.issuer.isin(default_bond.issuer.unique())]
def get_last_data(item, data):
    tmp_data = data[data.issuer == item['issuer']]
    row = len(tmp_data)
    if row == 0:
        return None
    for i in range(row):
        if item['default_date'] < tmp_data.iloc[i]['issue_date']:
            try:
                return tmp_data.iloc[i-1]
            except:
                return None
    return tmp_data.iloc[row-1]

tmp_data = default_bond.apply(get_last_data, axis=1, data = default_finance_data).\
            drop(['issuer'], axis=1)
default_bond = pd.concat([default_bond,tmp_data], axis=1).\
            drop(['default_date'], axis=1)
default_bond = default_bond[~default_bond.issue_date.isna()].\
            drop(['issue_date'],axis=1)

#########################
##### 数据清洗
#########################
def process(data):
    tmp_data = data.copy()
    # 填补缺失值
    tmp_data['new_level'] = tmp_data['new_level'].fillna('N')
    tmp_data['issue_com_level'] = tmp_data['issue_com_level'].fillna('N')
    tmp_data['issue_level'] = tmp_data['issue_level'].fillna('N')
    tmp_data['com_new_level'] = tmp_data['com_new_level'].fillna('N')

    filter_con = ~tmp_data.iloc[:,0].isna()
    for _col in tmp_data.columns:
        filter_con = filter_con & (~tmp_data[_col].isna())

    tmp_data = tmp_data[filter_con]

    for _col in ['start','end']:
        tmp_data[_col] = pd.to_datetime(tmp_data[_col]).apply(lambda x: x.year).astype(object)

    tmp_data.drop(['interest','price','issuer','end','bond_code'],axis=1,inplace=True)

    cols = list(tmp_data.columns)

    for _col in ['amount','capital','total_amount'] + \
                cols[cols.index('总资产(亿元)'):]:
        try:
            tmp_data[_col] = tmp_data[_col].astype(float)
        except:
            tmp_data[_col] = tmp_data[_col].apply(lambda x: float(re.sub(',', '', x)))
    return tmp_data

def print_object(default,normal):
    for _col in default.columns:
        if default[_col].dtype == object:
            print(f'{_col}--Normal Data:{normal[_col].nunique()}, '
                  f'Default Data:{default[_col].nunique()}.')


d = process(default_bond)
n = process(normal_bond)
d['label'] = 1
n['label'] = 0
data = pd.concat([d,n])
data = data.iloc[np.random.choice(range(len(data)),len(data),replace=False)].reset_index(drop=True)

train_data, test_data, _, _ = train_test_split(data, data.label,
                                               test_size = .2, random_state=1994)
train_data.to_pickle('train_data.pkl')
test_data.to_pickle('test_data.pkl')

