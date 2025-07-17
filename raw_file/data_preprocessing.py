
import pandas as pd
import warnings
import numpy as np

warnings.filterwarnings('ignore')

def preprocess(df):
    # コピーを作成して元のDataFrameを変更しないようにする
    df_processed = df.copy()

    # 1. 築年数 (age) から数値（年数）を抽出
    df_processed['age_years'] = df_processed['age'].str.extract(r'(\d+)').astype(float)
    # 新築の場合は0年とする
    df_processed.loc[df_processed['age'] == '新築', 'age_years'] = 0

    # 2. 建物階数 (floors) から数値（地上階数）を抽出
    df_processed['total_floors'] = df_processed['floors'].str.extract(r'(\d+)階建').astype(float)

    # 3. 交通アクセス (transportation) から駅徒歩分数を抽出 (最も近いもの)
    walk_minutes = df_processed['transportation_1'].str.extract(r'歩(\d+)分').astype(float)
    df_processed['walk_minutes'] = walk_minutes

    # 4. 面積 (area) を数値に変換
    df_processed['area_m2'] = df_processed['area'].astype(float)

    # 5. 間取り (layout) をダミー変数に変換
    df_processed['has_L'] = df_processed['layout'].str.contains('L').astype(int)
    df_processed['has_D'] = df_processed['layout'].str.contains('D').astype(int)
    df_processed['has_K'] = df_processed['layout'].str.contains('K').astype(int)
    df_processed['has_S'] = df_processed['layout'].str.contains('S').astype(int)
    df_processed['has_R'] = df_processed['layout'].str.contains('R').astype(int)
    df_processed['layout_rooms'] = df_processed['layout'].str.extract(r'(\d+)').astype(float)
    df_processed['layout_rooms'].fillna(1, inplace=True)

    # 6. 住所(address)から市区町村を抽出
    df_processed['city'] = df['address'].str.extract(r'東京都(.*?[市区])')
    df_processed['city'].fillna('不明', inplace=True)

    # 7. 交通アクセス(transportation_1)から路線名を抽出
    df_processed['line'] = df['transportation_1'].str.split('/').str[0]
    df_processed['line'].fillna('不明', inplace=True)

    # 8. カテゴリカル変数をOne-Hotエンコーディング
    categorical_features = ['city', 'line']
    df_processed = pd.get_dummies(df_processed, columns=categorical_features, dummy_na=False)

    # 9. 不要な列を削除
    df_processed = df_processed.drop([
        'building_name', 'address', 'transportation_1', 'transportation_2',
        'transportation_3', 'age', 'floors', 'layout', 'area'
    ], axis=1)

    # 10. 欠損値の処理
    for col in ['age_years', 'total_floors', 'walk_minutes']:
        df_processed[col].fillna(df_processed[col].median(), inplace=True)

    # rent_logを追加 (モデリング用)
    df_processed['rent_log'] = np.log1p(df['rent'])

    return df_processed

def main():
    # データを読み込む
    df = pd.read_csv('data/suumo_data.csv')

    # 前処理を実行
    df_clean = preprocess(df)

    # 前処理済みのデータを保存
    output_path = 'data/suumo_data_cleaned.csv'
    df_clean.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f'Preprocessed data saved to {output_path}')

if __name__ == '__main__':
    main()
