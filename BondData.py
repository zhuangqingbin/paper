import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


#########################
##### 读取债券数据
#########################
bond_drop_cols = ['Unnamed: 0','row_names','security_type','call_price',
                  'maturity_embedded', 'latest_date','call_content',
                  'next_interest_payment','rating_date','type_id', 'fullname']
bond_data = pd.read_csv('bond_data/static_bond.csv').drop(bond_drop_cols,axis=1)
# level_name/special_conditon缺失较为严重，标记为N
bond_data.level_name = bond_data.level_name.fillna('N')
bond_data.special_conditon = bond_data.special_conditon.fillna('N')
bond_data.listed_date = pd.to_datetime(bond_data.listed_date).apply(lambda x : str(x.year))
bond_data.delisted_date = pd.to_datetime(bond_data.delisted_date).apply(lambda x : str(x.year))


#########################
##### 读取企业基本信息数据
#########################
institution_drop_cols = ['Unnamed: 0','row_names','fullname','country','address',
                        'office','zipecode','phone','fax','email','website',
                         'social_code','institution_code','representitive','ceo',
                         'discloser','end_date','brief','industry_code','nature',
                         'ann_date','guarantor','amount','total_guarantee_amount',
                         'inward','outward','guarantee_bond','rating_date','scope',
                         'main_business','level1_id','level2_id']
static_institution = pd.read_csv('bond_data/static_institution.csv').drop(institution_drop_cols,axis=1)

static_institution = static_institution[static_institution.currency=='CNY'].drop(['currency'],axis=1)

static_institution.fount_date = pd.to_datetime(static_institution.fount_date).apply(lambda x : str(x.year))
static_institution.nature_id = static_institution.nature_id.astype(object)
static_institution.is_finance = static_institution.is_finance.astype(object)
static_institution.is_listed = static_institution.is_listed.astype(object)


#########################
##### 读取企业财务信息数据
#########################
basic_indic = pd.read_csv('bond_data/basic_indic.csv')

indic_drop_cols = ['Unnamed: 0','id','end_date']
indic_value = pd.read_csv('bond_data/indic_value.csv').drop(indic_drop_cols,axis=1)
use_indics = ['A0060','A0111','A0123',
              'B0001','B0022','B0041','B0045','B0048',
              'C0015','C0025','C0032','C0038','C0045','C0050']
indic_value = indic_value[indic_value.statement_typecode==1]
indic_value = indic_value[indic_value.indic_id.isin(use_indics)].\
    drop(['unit','industry_code','indic_v_belong_id','opdate',
          'user_id','etl_src','statement_typecode'],axis=1)
indic_value_lastest = indic_value[(indic_value.indic_v_year==2017) & (indic_value.indic_v_type==2)].\
    drop(['indic_v_year','indic_v_type'],axis=1)
institution_fin = indic_value_lastest.pivot(index='institution_id',columns='indic_id',
                                            values='indic_value').reset_index()

#########################
##### 合并为最终数据
#########################
final_data = pd.merge(bond_data,
           pd.merge(static_institution, institution_fin,on ='institution_id',how='left'),
           on = 'institution_id',how='left')

model_data = final_data.drop(['security_id','main_id','institution_id','trade_code',
                              'city','guarantee_rate','net_asset','revenue','amount'],axis=1).\
                            rename({'default':'label'},axis=1)

filter_con = (~model_data.C0045.isna()) & (~model_data.C0050.isna()) & (~model_data.C0032.isna())
for _col in ['security_type_name', 'label', 'residual_day', 'residual_maturity',
       'level_name_x', 'special_conditon', 'listed_date', 'delisted_date',
       'exchange_market', 'type_name', 'default_num', 'province', 'capital',
       'fount_date', 'comp_type', 'industry_name', 'nature_id', 'is_finance',
       'is_listed', 'level_name_y', 'warning_state']:
    filter_con = filter_con & (~model_data[_col].isna())
model_data = model_data[filter_con | model_data.label==1]
for _col in ['default_num','capital']:
    model_data[_col] = model_data[_col].fillna(np.mean(model_data[_col]))
for _col in ['province','comp_type','industry_name','nature_id','fount_date',
             'is_finance','is_listed','level_name_y','warning_state']:
    model_data[_col] = model_data[_col].fillna('未知')
for _col in ['A0060', 'A0111', 'A0123', 'B0001', 'B0022',
             'B0041', 'B0045', 'B0048', 'C0015', 'C0025',
             'C0032', 'C0038', 'C0045', 'C0050']:
    model_data[_col] = model_data[_col].fillna(method='ffill')
model_data['fount_date'] = model_data['fount_date'].apply(lambda x: x if x.startswith('2') else '2000')

train_data, test_data, _, _ = train_test_split(model_data, model_data.label,
                                               test_size = .2, random_state=1994)
train_data.to_pickle('train_data.pkl')
test_data.to_pickle('test_data.pkl')

#########################
##### 最终数据分析
#########################
# for _col in model_data.columns:
#     if model_data[_col].dtype == object:
#         print(f'{_col}:{model_data[_col].nunique()}')
#
# model_data.label.value_counts()