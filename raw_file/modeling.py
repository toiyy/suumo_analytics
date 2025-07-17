
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # 前処理済みのデータを読み込む
    df = pd.read_csv('data/suumo_data_cleaned.csv')

    # 特徴量 (X) とターゲット (y) を定義
    # rent_log をターゲットとし、元のrentは含めない
    # 元のrentカラムがdfにない場合があるため、存在チェックを行う
    if 'rent' in df.columns:
        X = df.drop(['rent', 'rent_log'], axis=1)
    else:
        # rentがない場合はrent_logのみをドロップ
        X = df.drop(['rent_log'], axis=1)

    y = df['rent_log']

    # 訓練データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f'訓練データのサイズ: {X_train.shape}')
    print(f'テストデータのサイズ: {X_test.shape}')

    # LightGBMモデルの学習
    lgb_reg = lgb.LGBMRegressor(random_state=42)
    lgb_reg.fit(X_train, y_train)

    # テストデータで予測
    y_pred_log_lgb = lgb_reg.predict(X_test)

    # 予測結果を元のスケールに戻す (log1pの逆変換はexpm1)
    y_pred_lgb = np.expm1(y_pred_log_lgb)
    y_test_orig = np.expm1(y_test)

    # RMSE (Root Mean Squared Error) で評価
    rmse_lgb = np.sqrt(mean_squared_error(y_test_orig, y_pred_lgb))
    print(f'LightGBM RMSE (New Features): {rmse_lgb:.4f} (万円)')

    # 特徴量の重要度を可視化
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': lgb_reg.feature_importances_
    }).sort_values('importance', ascending=False)

    # 重要度が高い特徴量のみをプロット (上位20件)
    plt.figure(figsize=(10, 10))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
    plt.title('特徴量の重要度 (LightGBM, New Features)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print('Saved feature importance plot to feature_importance.png')

if __name__ == '__main__':
    main()
