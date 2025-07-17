

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore', category=UserWarning) # Suppress UserWarning from matplotlib/seaborn

def main():
    # 前処理済みのデータを読み込む
    df = pd.read_csv('data/suumo_data_cleaned.csv')

    # 特徴量 (X) とターゲット (y) を定義
    if 'rent' in df.columns:
        X = df.drop(['rent', 'rent_log'], axis=1)
    else:
        X = df.drop(['rent_log'], axis=1)

    y = df['rent_log']

    # 交差検証の設定
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    rmse_scores = []

    print(f'Starting {n_splits}-fold cross-validation...')

    for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
        print(f'--- Fold {fold+1}/{n_splits} ---')

        # 訓練データと検証データに分割
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # LightGBMモデルの学習
        lgb_reg = lgb.LGBMRegressor(random_state=42)
        lgb_reg.fit(X_train, y_train)

        # 検証データで予測
        y_pred_log = lgb_reg.predict(X_val)

        # 予測結果を元のスケールに戻す
        y_pred = np.expm1(y_pred_log)
        y_val_orig = np.expm1(y_val)

        # RMSEで評価
        rmse = np.sqrt(mean_squared_error(y_val_orig, y_pred))
        rmse_scores.append(rmse)
        print(f'RMSE for fold {fold+1}: {rmse:.4f} (万円)')

    # 平均RMSEを計算
    mean_rmse = np.mean(rmse_scores)
    std_rmse = np.std(rmse_scores)
    print('\n--- Cross-Validation Summary ---')
    print(f'Mean RMSE: {mean_rmse:.4f} (万円)')
    print(f'Std Dev of RMSE: {std_rmse:.4f} (万円)')

if __name__ == '__main__':
    main()

