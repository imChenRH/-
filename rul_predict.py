# -*- coding: utf-8 -*-
"""
发动机剩余寿命预测
基于随机森林和梯度提升的集成学习方法
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 检查是否安装了xgboost和lightgbm
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


# 数据加载函数
def load_data(filepath):
    """读取Excel数据文件"""
    cols = ['drop', 'unit_id', 'time_cycle', 'setting_1', 'setting_2', 'setting_3',
            's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10',
            's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
    data = pd.read_excel(filepath, skiprows=2, names=cols)
    return data.drop('drop', axis=1)


# 计算剩余寿命
def calc_rul(data):
    """为每条记录计算RUL值"""
    max_cyc = data.groupby('unit_id')['time_cycle'].transform('max')
    data['RUL'] = max_cyc - data['time_cycle']
    return data


# 特征提取
def extract_features(data, sensor_list):
    """从传感器数据中提取统计特征"""
    data = data.sort_values(['unit_id', 'time_cycle']).reset_index(drop=True)
    
    for sensor in sensor_list:
        # 移动平均
        data[sensor + '_ma5'] = data.groupby('unit_id')[sensor].transform(
            lambda x: x.rolling(5, min_periods=1).mean())
        data[sensor + '_ma10'] = data.groupby('unit_id')[sensor].transform(
            lambda x: x.rolling(10, min_periods=1).mean())
        # 变化量
        data[sensor + '_delta'] = data.groupby('unit_id')[sensor].transform(
            lambda x: x.diff().fillna(0))
        # 累计偏差
        data[sensor + '_dev'] = data.groupby('unit_id')[sensor].transform(
            lambda x: x - x.iloc[0])
    
    # 周期对数
    data['log_cycle'] = np.log1p(data['time_cycle'])
    return data


# 安全评分函数
def safety_score(y_real, y_hat):
    """计算安全惩罚分数"""
    diff = y_hat - y_real
    score = np.where(diff < 0, np.exp(-diff/13) - 1, np.exp(diff/10) - 1)
    return np.sum(score)


# 模型训练与预测
def run_prediction(train_path, test_path, name, rul_max=125):
    """训练模型并预测"""
    print('\n' + '='*40)
    print(name)
    print('='*40)
    
    # 读数据
    train_df = load_data(train_path)
    test_df = load_data(test_path)
    
    # 计算RUL
    train_df = calc_rul(train_df)
    
    # 筛选有效传感器
    valid_sensors = []
    for i in range(1, 22):
        col = 's' + str(i)
        if train_df[col].std() > 0.01:
            r = abs(train_df[col].corr(train_df['RUL']))
            if r > 0.1:
                valid_sensors.append(col)
    valid_sensors = valid_sensors[:10]
    print('选用传感器:', valid_sensors)
    
    # 特征提取
    train_df = extract_features(train_df, valid_sensors)
    test_df = extract_features(test_df, valid_sensors)
    
    # RUL上限截断
    train_df['RUL_cap'] = train_df['RUL'].clip(upper=rul_max)
    
    # 准备训练集
    skip_cols = ['unit_id', 'time_cycle', 'RUL', 'RUL_cap']
    feat_cols = [c for c in train_df.columns if c not in skip_cols]
    
    X = train_df[feat_cols].fillna(0)
    y = train_df['RUL_cap']
    print('特征维度:', len(feat_cols), '样本数:', len(X))
    
    # 构建模型
    print('训练中...')
    model_list = []
    
    # 随机森林
    m1 = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    m1.fit(X, y)
    model_list.append((m1, 0.35))
    
    # 梯度提升树
    m2 = GradientBoostingRegressor(n_estimators=80, max_depth=8, learning_rate=0.1, random_state=42)
    m2.fit(X, y)
    model_list.append((m2, 0.25))
    
    # XGBoost（如果可用）
    if HAS_XGB:
        m3 = XGBRegressor(n_estimators=80, max_depth=8, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0)
        m3.fit(X, y)
        model_list.append((m3, 0.20))
    
    # LightGBM（如果可用）
    if HAS_LGBM:
        m4 = LGBMRegressor(n_estimators=80, max_depth=8, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
        m4.fit(X, y)
        model_list.append((m4, 0.20))
    
    # 权重归一化
    w_sum = sum(w for _, w in model_list)
    model_list = [(m, w/w_sum) for m, w in model_list]
    
    # 模型验证
    val_df = train_df.groupby('unit_id').last().reset_index()
    X_val = val_df[feat_cols].fillna(0)
    y_val = val_df['RUL_cap']
    
    pred_val = sum(m.predict(X_val) * w for m, w in model_list)
    err = np.sqrt(mean_squared_error(y_val, pred_val))
    ss = safety_score(y_val.values, pred_val)
    print('验证误差 RMSE:', round(err, 4), ' 安全分:', round(ss, 2))
    
    # 测试集预测
    test_last = test_df.groupby('unit_id').last().reset_index()
    for col in feat_cols:
        if col not in test_last.columns:
            test_last[col] = 0
    
    X_test = test_last[feat_cols].fillna(0)
    pred = sum(m.predict(X_test) * w for m, w in model_list)
    
    # 保守修正
    pred = pred - 2
    pred = np.clip(np.round(pred), 0, rul_max).astype(int)
    
    print('预测范围:', pred.min(), '-', pred.max(), '均值:', round(pred.mean(), 1))
    
    out = pd.DataFrame({'unit_id': test_last['unit_id'], 'RUL': pred})
    return out.sort_values('unit_id')['RUL'].values


# 主程序
if __name__ == '__main__':
    import os
    work_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 问题一
    result1 = run_prediction(
        os.path.join(work_dir, '题1_train.xlsx'),
        os.path.join(work_dir, '题1_test.xlsx'),
        '问题一: HPC性能退化预测',
        125
    )
    
    # 问题二
    result2 = run_prediction(
        os.path.join(work_dir, '题2_train.xlsx'),
        os.path.join(work_dir, '题2_test.xlsx'),
        '问题二: 多故障模式预测',
        150
    )
    
    # 输出结果
    output = pd.DataFrame({
        '问题1_RUL': result1,
        '问题2_RUL': result2
    })
    output.to_excel(os.path.join(work_dir, '提交结果.xlsx'), index=False)
    print('\n完成! 结果保存至 提交结果.xlsx')
