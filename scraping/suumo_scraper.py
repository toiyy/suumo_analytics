import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from tqdm import tqdm
import re
import os

# SUUMOの検索結果URLのテンプレート（賃料範囲を指定可能に）
BASE_URL_TEMPLATE = "https://suumo.jp/jj/chintai/ichiran/FR301FC001/?ar=030&bs=040&ta={pref_code}&sc={area_code}&cb={min_rent}&ct={max_rent}&mb=0&mt=9999999&et=9999999&cn=9999999&shkr1=03&shkr2=03&shkr3=03&shkr4=03&sngz=&po1=25&pc=50"

# 収集対象のエリアリスト
AREAS = [
    ("渋谷区", "13", "13113"),
    ("新宿区", "13", "13104"),
    ("港区", "13", "13103"),
    ("世田谷区", "13", "13112"),
    ("目黒区", "13", "13110"),
    ("品川区", "13", "13109"),
    ("横浜市", "14", "14100"),
]

# 収集対象の賃料範囲リスト (下限, 上限) 単位：万円
RENT_RANGES = [(i, i + 2) for i in range(5, 30, 2)] # 5-7万, 7-9万, ..., 29-31万

def get_html(url):
    """指定されたURLからHTMLコンテンツを取得する"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        time.sleep(2) # サーバー負荷軽減のため2秒待機
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        return response.text
    except requests.exceptions.RequestException as e:
        # print(f"\nError fetching {url}: {e}") # ログが多すぎるのでコメントアウト
        return None

def parse_detail_page(html):
    """詳細ページから追加情報を抽出する"""
    soup = BeautifulSoup(html, 'lxml')
    details = {}
    property_view_table = soup.find('table', class_='property_view_table')
    if property_view_table:
        rows = property_view_table.find_all('tr')
        for row in rows:
            th = row.find('th')
            td = row.find('td')
            if th and td:
                header = th.text.strip()
                if '構造' in header:
                    details['structure'] = td.text.strip()
                elif '階' in header and '階建' in header:
                    floor_info = td.text.strip()
                    if '/' in floor_info:
                        details['floor_number'] = floor_info.split('/')[1]
                    else:
                        details['floor_number'] = floor_info
                elif '向き' in header:
                    details['direction'] = td.text.strip()
    features_html = str(soup)
    details['has_separate_bath_toilet'] = 1 if 'バス・トイレ別' in features_html else 0
    details['has_reheating'] = 1 if '追焚機能' in features_html else 0
    details['has_bathroom_dryer'] = 1 if '浴室乾燥機' in features_html else 0
    details['has_autolock'] = 1 if 'オートロック' in features_html else 0
    details['has_tv_intercom'] = 1 if 'TVモニタ付インターホン' in features_html else 0
    details['has_delivery_box'] = 1 if '宅配ボックス' in features_html else 0
    details['has_pet_allowed'] = 1 if 'ペット相談' in features_html else 0
    details['has_musical_instruments_allowed'] = 1 if '楽器相談' in features_html else 0
    details['has_free_internet'] = 1 if 'インターネット無料' in features_html else 0
    details['has_system_kitchen'] = 1 if 'システムキッチン' in features_html else 0
    details['has_gas_stove_gt2'] = 1 if 'コンロ二口以上' in features_html else 0
    details['is_top_floor'] = 1 if '最上階' in features_html else 0
    details['is_corner_room'] = 1 if '角部屋' in features_html else 0
    return details

def parse_properties(html, pbar_details):
    """HTMLから物件情報のリストを抽出する"""
    soup = BeautifulSoup(html, 'lxml')
    properties = []
    cassette_items = soup.find_all('div', class_='cassetteitem')
    for item in cassette_items:
        building_name = item.find('div', class_='cassetteitem_content-title').text.strip()
        address = item.find('li', class_='cassetteitem_detail-col1').text.strip()
        transportation_elements = item.find_all('div', class_='cassetteitem_detail-text')
        transportations = [t.text.strip() for t in transportation_elements]
        age_and_floors = item.find('li', class_='cassetteitem_detail-col3').find_all('div')
        age = age_and_floors[0].text.strip()
        floors = age_and_floors[1].text.strip()
        table_rows = item.find_all('tr', class_='js-cassette_link')
        for row in table_rows:
            try:
                detail_url_relative = row.find('a', class_='js-cassette_link_href')['href']
                detail_url = "https://suumo.jp" + detail_url_relative
                rent_text = row.find('span', class_='cassetteitem_price--rent').text.strip()
                rent = float(re.sub(r'[^\d.]', '', rent_text))
                admin_fee_text = row.find('span', class_='cassetteitem_price--administration').text.strip()
                admin_fee = 0 if admin_fee_text == '-' else int(re.sub(r'[^\d]', '', admin_fee_text))
                deposit_text = row.find('span', class_='cassetteitem_price--deposit').text.strip()
                deposit = 0 if deposit_text == '-' else float(re.sub(r'[^\d.]', '', deposit_text))
                gratuity_text = row.find('span', class_='cassetteitem_price--gratuity').text.strip()
                gratuity = 0 if gratuity_text == '-' else float(re.sub(r'[^\d.]', '', gratuity_text))
                layout = row.find('span', class_='cassetteitem_madori').text.strip()
                area = row.find('span', class_='cassetteitem_menseki').text.strip().replace('m2', '')
                property_data = {
                    'building_name': building_name, 'address': address,
                    'transportation_1': transportations[0] if len(transportations) > 0 else None,
                    'transportation_2': transportations[1] if len(transportations) > 1 else None,
                    'transportation_3': transportations[2] if len(transportations) > 2 else None,
                    'age': age, 'floors': floors, 'rent': rent, 'admin_fee': admin_fee,
                    'deposit': deposit, 'gratuity': gratuity, 'layout': layout, 'area': area,
                    'detail_url': detail_url
                }
                detail_html = get_html(detail_url)
                if detail_html:
                    pbar_details.update(1)
                    additional_details = parse_detail_page(detail_html)
                    property_data.update(additional_details)
                properties.append(property_data)
            except (AttributeError, ValueError, TypeError):
                continue
    return properties

def get_next_page_url(html):
    """次のページのURLを取得する"""
    soup = BeautifulSoup(html, 'lxml')
    next_page_container = soup.find('p', class_='pager_next')
    if next_page_container:
        next_page_tag = next_page_container.find('a')
        if next_page_tag and 'href' in next_page_tag.attrs:
            return "https://suumo.jp" + next_page_tag['href']
    return None

if __name__ == "__main__":
    output_path = 'data/suumo_data_final.csv'
    total_properties = []

    # 賃料範囲でループ
    for min_rent, max_rent in tqdm(RENT_RANGES, desc="Overall Rent Progress"):
        # エリアでループ
        for area_name, pref_code, area_code in AREAS:
            start_url = BASE_URL_TEMPLATE.format(pref_code=pref_code, area_code=area_code, min_rent=min_rent, max_rent=max_rent)
            current_url = start_url
            
            progress_desc = f"{area_name} ({min_rent}-{max_rent}万)"
            with tqdm(desc=progress_desc, leave=False) as pbar_details:
                while current_url:
                    html = get_html(current_url)
                    if not html:
                        break
                    
                    properties = parse_properties(html, pbar_details)
                    if not properties:
                        break # そのページに物件がなければ終了
                    
                    total_properties.extend(properties)
                    current_url = get_next_page_url(html)

    if total_properties:
        df = pd.DataFrame(total_properties)
        # データを追記モードで保存
        # ファイルが存在しない場合はヘッダーを書き込み、存在する場合は追記のみ
        df.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False, encoding='utf-8-sig')
        
        # 最後に全体の重複を削除
        df_final = pd.read_csv(output_path)
        df_final.drop_duplicates(inplace=True)
        df_final.to_csv(output_path, index=False, encoding='utf-8-sig')

        print(f"\n\nSuccessfully finished scraping.")
        print(f"A total of {len(df_final)} unique properties were saved to {output_path}")
    else:
        print("No properties were scraped.")