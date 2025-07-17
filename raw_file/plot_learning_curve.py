
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import learning_curve, KFold
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

def plot_learning_curve_rmse(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("RMSE (万円)")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring="neg_mean_squared_error")

    # スコアは負のMSEで返ってくるので、RMSEに変換
    train_rmse = np.sqrt(-train_scores)
    test_rmse = np.sqrt(-test_scores)

    train_rmse_mean = np.mean(train_rmse, axis=1)
    train_rmse_std = np.std(train_rmse, axis=1)
    test_rmse_mean = np.mean(test_rmse, axis=1)
    test_rmse_std = np.std(test_rmse, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_rmse_mean - train_rmse_std,
                     train_rmse_mean + train_rmse_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_rmse_mean - test_rmse_std,
                     test_rmse_mean + test_rmse_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_rmse_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_rmse_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.tight_layout()
    return plt

def main():
    # データを読み込む
    df = pd.read_csv('data/suumo_data_cleaned.csv')

    # 特徴量 (X) とターゲット (y) を定義
    if 'rent' in df.columns:
        X = df.drop(['rent', 'rent_log'], axis=1)
    else:
        X = df.drop(['rent_log'], axis=1)

    y = df['rent_log']

    # モデルを定義
    estimator = lgb.LGBMRegressor(random_state=42)

    # 交差検証の設定
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # 学習曲線を描画
    title = "Learning Curves (LightGBM)"
    plot = plot_learning_curve_rmse(estimator, title, X, y, cv=cv, n_jobs=-1)
    
    # グラフをファイルに保存
    output_path = 'learning_curve.png'
    plot.savefig(output_path)
    print(f'Learning curve plot saved to {output_path}')

if __name__ == '__main__':
    main()
