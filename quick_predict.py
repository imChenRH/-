"""
飞机发动机RUL预测 - 快速版本
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    """加载数据"""
    col_names = ['drop', 'unit_id', 'time_cycle', 'setting_1', 'setting_2', 'setting_3',
                 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10',
                 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
    df = pd.read_excel(file_path, skiprows=2, names=col_names)
    return df.drop('drop', axis=1)

def add_rul(df):
    """添加RUL列"""
    max_cycles = df.groupby('unit_id')['time_cycle'].max()
    df['RUL'] = df.apply(lambda x: max_cycles[x['unit_id']] - x['time_cycle'], axis=1)
    return df

def add_features(df):
    """简单特征工程"""
    sensors = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
    
    for col in sensors:
        df[f'{col}_rm5'] = df.groupby('unit_id')[col].transform(lambda x: x.rolling(5, min_periods=1).mean())
        df[f'{col}_diff'] = df.groupby('unit_id')[col].transform(lambda x: x.diff().fillna(0))
    
    return df

def solve(train_file, test_file, max_rul=125):
    """解决问题"""
    # 加载数据
    df_train = load_data(train_file)
    df_test = load_data(test_file)
    
    # 添加RUL和特征
    df_train = add_rul(df_train)
    df_train = add_features(df_train)
    df_test = add_features(df_test)
    
    # RUL上限
    df_train['RUL'] = df_train['RUL'].clip(upper=max_rul)
    
    # 特征列
    exclude = ['unit_id', 'time_cycle', 'RUL']
    features = [c for c in df_train.columns if c not in exclude]
    
    # 训练
    X_train = df_train[features].fillna(0)
    y_train = df_train['RUL']
    
    model = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # 预测（每个发动机最后一条记录）
    last_test = df_test.groupby('unit_id').last().reset_index()
    X_test = last_test[features].fillna(0)
    
    y_pred = model.predict(X_test)
    y_pred = np.maximum(np.round(y_pred), 0).astype(int)
    
    results = pd.DataFrame({'unit_id': last_test['unit_id'], 'RUL': y_pred})
    return results.sort_values('unit_id')['RUL'].values

def main():
    base = "/home/runner/work/-/-"
    
    print("解决问题1...")
    rul1 = solve(f"{base}/题1_train.xlsx", f"{base}/题1_test.xlsx", 125)
    print(f"问题1完成: {rul1[:5]}...")
    
    print("解决问题2...")
    rul2 = solve(f"{base}/题2_train.xlsx", f"{base}/题2_test.xlsx", 150)
    print(f"问题2完成: {rul2[:5]}...")
    
    # 保存结果
    pd.DataFrame({'问题1_RUL': rul1, '问题2_RUL': rul2}).to_excel(f"{base}/提交结果.xlsx", index=False)
    print("结果已保存!")

if __name__ == "__main__":
    main()
