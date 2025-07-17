

import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

def find_bargains():
    # データを読み込む
    try:
        df_clean = pd.read_csv('data/suumo_data_cleaned.csv')
        df_raw = pd.read_csv('data/suumo_data.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}. Please make sure the data files exist.")
        return

    # 特徴量 (X) とターゲット (y) を定義
    if 'rent' not in df_clean.columns or 'rent_log' not in df_clean.columns:
        print("Error: 'rent' or 'rent_log' column not found in cleaned data.")
        return
        
    X = df_clean.drop(['rent', 'rent_log'], axis=1)
    y = df_clean['rent_log']

    print("Training model on the entire dataset...")
    # LightGBMモデルをデータ全体で学習
    lgb_reg = lgb.LGBMRegressor(random_state=42)
    lgb_reg.fit(X, y)

    print("Predicting rent for all properties...")
    # 全データの家賃を予測
    all_pred_log = lgb_reg.predict(X)
    all_pred = np.expm1(all_pred_log)

    # 元のデータに予測結果を結合
    df_result = df_raw.copy()
    df_result['actual_rent'] = df_clean['rent']
    df_result['predicted_rent'] = all_pred
    df_result['difference'] = df_result['actual_rent'] - df_result['predicted_rent']
    df_result['discount_rate'] = df_result['difference'] / df_result['actual_rent']

    # 割安度の高い順にソート
    df_bargain = df_result.sort_values("discount_rate", ascending=True)

    # 結果の表示
    print("\n--- 割安物件ランキング TOP 20 (新モデル) ---")
    
    # 表示する列を定義
    display_columns = [
        'address', 'building_name', 'age', 'floors', 'layout', 'area',
        'actual_rent', 'predicted_rent', 'discount_rate'
    ]
    
    # カラムの存在を確認
    display_columns = [col for col in display_columns if col in df_bargain.columns]

    # pandasの表示オプションを設定して、すべての列を表示
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print(df_bargain[display_columns].head(20).to_string(index=False))

if __name__ == '__main__':
    find_bargains()

