import gc

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from xgboost import XGBClassifier

paths = './data_format1'
data = pd.read_csv(f'{paths}/user_log_format1.csv', dtype={'time_stamp':'str'})
data1 = pd.read_csv(f'{paths}/user_info_format1.csv')
data2 = pd.read_csv(f'{paths}/train_format1.csv')
submission = pd.read_csv(f'{paths}/test_format1.csv')
data_train = pd.read_csv('./data_format2/train_format2.csv')

data2['origin'] = 'train'
submission['origin'] = 'test'
matrix = pd.concat([data2, submission], ignore_index=True, sort=False)
matrix.drop(['prob'], axis=1, inplace=True)
matrix = matrix.merge(data1, on='user_id', how='left')
data.rename(columns={'seller_id':'merchant_id'}, inplace=True)

data['user_id'] = data['user_id'].astype('int32')
data['merchant_id'] = data['merchant_id'].astype('int32')
data['item_id'] = data['item_id'].astype('int32')
data['cat_id'] = data['cat_id'].astype('int32')
data['brand_id'].fillna(0, inplace=True)
data['brand_id'] = data['brand_id'].astype('int32')
data['time_stamp'] = pd.to_datetime(data['time_stamp'], format='%H%M')

matrix['age_range'].fillna(0, inplace=True)
matrix['gender'].fillna(2, inplace=True)
matrix['age_range'] = matrix['age_range'].astype('int8')
matrix['gender'] = matrix['gender'].astype('int8')
matrix['label'] = matrix['label'].astype('str')
matrix['user_id'] = matrix['user_id'].astype('int32')
matrix['merchant_id'] = matrix['merchant_id'].astype('int32')

del data1, data2
gc.collect()

#特征处理
#数据集 data 按照用户ID (user_id) 进行分组，创建了一个分组对象 groups。这个对象可以用来对每个用户的行为进行聚合分析
groups = data.groupby(['user_id'])

#groups.size() 计算每个分组（即每个用户）的行为总数。
#reset_index() 将分组的大小转换为一个DataFrame，并将分组键（user_id）设置为DataFrame的一列。
#rename(columns={0:'u1'}) 将计数列的名称从默认的0改为更具描述性的 ‘u1’。
temp = groups.size().reset_index().rename(columns={0:'u1'})


#这一行代码将刚刚计算出的用户行为总数 temp 合并回主特征矩阵 matrix。合并是基于 user_id 列进行的，使用左连接（how='left'）确保所有用户都在合并后的矩阵中。
matrix = matrix.merge(temp, on='user_id', how='left')

#groups['item_id'].agg([('u2', 'nunique')]) 计算每个用户所交互的不同商品ID的数量。
#reset_index() 将这个聚合结果转换为一个DataFrame。
temp = groups['item_id'].agg([('u2', 'nunique')]).reset_index()

#以下针对不同列重复上述操作
matrix = matrix.merge(temp, on='user_id', how='left')
temp = groups['cat_id'].agg([('u3', 'nunique')]).reset_index()
matrix = matrix.merge(temp, on='user_id', how='left')
temp = groups['merchant_id'].agg([('u4', 'nunique')]).reset_index()
matrix = matrix.merge(temp, on='user_id', how='left')
temp = groups['brand_id'].agg([('u5', 'nunique')]).reset_index()
matrix = matrix.merge(temp, on='user_id', how='left')
temp = groups['time_stamp'].agg([('F_time', 'min'), ('L_time', 'max')]).reset_index()
temp['u6'] = (temp['L_time'] - temp['F_time']).dt.seconds/3600
matrix = matrix.merge(temp[['user_id', 'u6']], on='user_id', how='left')
temp = groups['action_type'].value_counts().unstack().reset_index().rename(columns={0:'u7', 1:'u8', 2:'u9', 3:'u10'})
matrix = matrix.merge(temp, on='user_id', how='left')

#这一行代码将数据集 data 按照商家ID (merchant_id) 进行分组，创建了一个分组对象 groups。这个对象可以用来对每个商家的行为进行聚合分析。
groups = data.groupby(['merchant_id'])

#groups.size() 计算每个分组（即每个商家）的行为总数。
#reset_index() 将分组的大小转换为一个DataFrame，并将分组键（merchant_id）设置为DataFrame的一列。
temp = groups.size().reset_index().rename(columns={0:'m1'})

#这一行代码将刚刚计算出的商家行为总数 temp 合并回主特征矩阵 matrix。合并是基于 merchant_id 列进行的，使用左连接（how='left'）确保所有商家都在合并后的矩阵中。
matrix = matrix.merge(temp, on='merchant_id', how='left')

#groups[['user_id', 'item_id', 'cat_id', 'brand_id']].nunique() 计算每个商家所交互的不同用户、商品、类别和品牌ID的数量。
#reset_index() 将这个聚合结果转换为一个DataFrame。
temp = groups[['user_id', 'item_id', 'cat_id', 'brand_id']].nunique().reset_index().rename(columns={
    'user_id':'m2',
    'item_id':'m3',
    'cat_id':'m4',
    'brand_id':'m5'})

#再次将计算出的特征 temp 合并回 matrix
matrix = matrix.merge(temp, on='merchant_id', how='left')

#value_counts() 计算每个行为类型的出现次数  unstack() 将计数转换为列  rename(columns={...}) 将列名重命名为更具描述性的名称。
temp = groups['action_type'].value_counts().unstack().reset_index().rename(columns={0:'m6', 1:'m7', 2:'m8', 3:'m9'})

#最后，将商家的行为类型计数合并回 matrix
matrix = matrix.merge(temp, on='merchant_id', how='left')

temp = data_train[data_train['label']==1].groupby(['merchant_id']).size().reset_index().rename(columns={0:'m10'})
matrix = matrix.merge(temp, on='merchant_id', how='left')

groups = data.groupby(['user_id', 'merchant_id'])
temp = groups.size().reset_index().rename(columns={0:'um1'})
matrix = matrix.merge(temp, on=['user_id', 'merchant_id'], how='left')
temp = groups[['user_id', 'item_id', 'cat_id', 'brand_id']] .nunique() .reset_index(drop=True).rename(columns={
           'user_id': 'm2',# 添加 drop=True 避免列名冲突
           'item_id': 'm3',
           'cat_id': 'm4',
           'brand_id': 'm5'
       })

#matrix = matrix.merge(temp, on=['user_id', 'merchant_id'], how='left')
temp = groups['action_type'].value_counts().unstack().reset_index().rename(columns={
    0:'um5',
    1:'um6',
    2:'um7',
    3:'um8'
})

#将之前计算出的特征 temp 合并回主特征矩阵 matrix。合并是基于 user_id 和 merchant_id 列进行的，使用左连接（how='left'）确保所有用户和商家对都在合并后的矩阵中。
matrix = matrix.merge(temp, on=['user_id', 'merchant_id'], how='left')
temp = groups['time_stamp'].agg([('frist', 'min'), ('last', 'max')]).reset_index()
temp['um9'] = (temp['last'] - temp['frist']).dt.seconds/3600
temp.drop(['frist', 'last'], axis=1, inplace=True)
matrix = matrix.merge(temp, on=['user_id', 'merchant_id'], how='left')

matrix['r1'] = matrix['u9']/matrix['u7'] #用户购买点击比
matrix['r2'] = matrix['m8']/matrix['m6'] #商家购买点击比
matrix['r3'] = matrix['um7']/matrix['um5'] #不同用户不同商家购买点击比

matrix.fillna(0, inplace=True)

temp = pd.get_dummies(matrix['age_range'], prefix='age')
matrix = pd.concat([matrix, temp], axis=1)
temp = pd.get_dummies(matrix['gender'], prefix='g')
matrix = pd.concat([matrix, temp], axis=1)
matrix.drop(['age_range', 'gender'], axis=1, inplace=True)

#train、test-setdata
train_data = matrix[matrix['origin'] == 'train'].drop(['origin'], axis=1)
test_data = matrix[matrix['origin'] == 'test'].drop(['label', 'origin'], axis=1)
train_X, train_y = train_data.drop(['label'], axis=1), train_data['label']

del temp, matrix
gc.collect()

''''#导入分析库
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb'''

# 将标签从字符串转换为整数
train_y = train_y.map({'0.0': 0, '1.0': 1}).astype(int)

# 直接训练XGBoost模型，移除未使用的RandomForestClassifier部分
xgb_model = XGBClassifier(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300,
    colsample_bytree=0.8,
    subsample=0.8,
    learning_rate=0.3,  # 使用learning_rate代替eta，因为eta已被弃用
    seed=42
)

xgb_model.fit(
    train_X,
    train_y,
    eval_metric='auc',
    eval_set=[(train_X, train_y)],
    verbose=True,
    early_stopping_rounds=20
)

# 预测并生成提交文件
prob = xgb_model.predict_proba(test_data)[:, 1]  # 取正类概率
submission['prob'] = prob
submission.drop(['origin'], axis=1, inplace=True)
submission.to_csv('submission.csv', index=False)
