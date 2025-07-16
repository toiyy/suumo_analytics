import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from tqdm import tqdm
import re

# SUUMOの検索結果URL（例: 東京都千代田区）
BASE_URL = "https://suumo.jp/jj/chintai/ichiran/FR301FC001/?ar=030&bs=040&ta=13&sc=13101&cb=0.0&ct=9999999&mb=0&mt=9999999&et=9999999&cn=9999999&shkr1=03&shkr2=03&shkr3=03&shkr4=03&sngz=&po1=25&pc=50"

def get_html(url):
    """指定されたURLからHTMLコンテンツを取得する"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def parse_properties(html):
    """HTMLから物件情報のリストを抽出する"""
    soup = BeautifulSoup(html, 'lxml')
    properties = []
    
    # 物件情報が格納されているコンテナ
    cassette_items = soup.find_all('div', class_='cassetteitem')

    for item in cassette_items:
        # 各部屋の情報を取得
        table_rows = item.find_all('tr', class_='js-cassette_link')
        
        # 建物共通の情報を取得
        building_name = item.find('div', class_='cassetteitem_content-title').text.strip()
        address = item.find('li', class_='cassetteitem_detail-col1').text.strip()
        
        # 交通アクセス情報を取得（複数ある場合があるのでリスト化）
        transportation_elements = item.find_all('div', class_='cassetteitem_detail-text')
        transportations = [t.text.strip() for t in transportation_elements]
        
        # 築年数と階数を取得
        age_and_floors = item.find('li', class_='cassetteitem_detail-col3').find_all('div')
        age = age_and_floors[0].text.strip()
        floors = age_and_floors[1].text.strip()

        for row in table_rows:
            try:
                rent_text = row.find('span', class_='cassetteitem_price--rent').text.strip()
                rent = float(re.sub(r'[^\d.]', '', rent_text)) # 万円を数値に

                admin_fee_text = row.find('span', class_='cassetteitem_price--administration').text.strip()
                admin_fee = 0 if admin_fee_text == '-' else int(re.sub(r'[^\d]', '', admin_fee_text))

                deposit_text = row.find('span', class_='cassetteitem_price--deposit').text.strip()
                deposit = 0 if deposit_text == '-' else float(re.sub(r'[^\d.]', '', deposit_text))

                gratuity_text = row.find('span', class_='cassetteitem_price--gratuity').text.strip()
                gratuity = 0 if gratuity_text == '-' else float(re.sub(r'[^\d.]', '', gratuity_text))

                layout = row.find('span', class_='cassetteitem_madori').text.strip()
                area = row.find('span', class_='cassetteitem_menseki').text.strip().replace('m2', '')

                property_data = {
                    'building_name': building_name,
                    'address': address,
                    'transportation_1': transportations[0] if len(transportations) > 0 else None,
                    'transportation_2': transportations[1] if len(transportations) > 1 else None,
                    'transportation_3': transportations[2] if len(transportations) > 2 else None,
                    'age': age,
                    'floors': floors,
                    'rent': rent,
                    'admin_fee': admin_fee,
                    'deposit': deposit,
                    'gratuity': gratuity,
                    'layout': layout,
                    'area': area,
                }
                properties.append(property_data)
            except (AttributeError, ValueError) as e:
                # print(f"Skipping a row due to parsing error: {e}")
                continue # パースエラーが発生した部屋はスキップ
                
    return properties

def get_next_page_url(html):
    """次のページのURLを取得する"""
    soup = BeautifulSoup(html, 'lxml')
    next_page_link = soup.find('a', text='次へ')
    if next_page_link and 'href' in next_page_link.attrs:
        return "https://suumo.jp" + next_page_link['href']
    return None

if __name__ == "__main__":
    all_properties = []
    current_url = BASE_URL
    page_count = 1

    # tqdmを使って進捗バーを表示
    pbar = tqdm(desc="Scraping pages")
    while current_url:
        print(f"Scraping page {page_count}: {current_url}")
        html = get_html(current_url)
        
        if not html:
            break

        properties = parse_properties(html)
        all_properties.extend(properties)
        
        current_url = get_next_page_url(html)
        page_count += 1
        pbar.update(1)
        
        # サーバーに負荷をかけすぎないように待機
        time.sleep(1)
        
        # テスト用に3ページで止める
        if page_count > 3:
            print("Stopping after 3 pages for testing.")
            break

    pbar.close()

    if all_properties:
        df = pd.DataFrame(all_properties)
        
        # 保存するファイル名を指定
        output_path = '../data/suumo_data.csv'
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Successfully scraped {len(all_properties)} properties.")
        print(f"Data saved to {output_path}")
        print("First 5 rows of the data:")
        print(df.head())
    else:
        print("No properties were scraped.")