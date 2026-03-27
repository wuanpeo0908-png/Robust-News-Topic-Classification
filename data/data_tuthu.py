import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
import csv

# Danh mục khớp với 10 nhãn của bạn
categories = {
   # 0: 'https://vnexpress.net/thoi-su',
    1: 'https://vnexpress.net/the-gioi',
    2: 'https://vnexpress.net/kinh-doanh/p',
    3: 'https://vnexpress.net/khoa-hoc',
    4: 'https://vnexpress.net/bat-dong-san',
    5: 'https://vnexpress.net/suc-khoe/p',
    6: 'https://vnexpress.net/the-thao/p',
    7: 'https://vnexpress.net/giai-tri/p',
    8: 'https://vnexpress.net/phap-luat',
    9: 'https://vnexpress.net/giao-duc'
}

save_path = r'D:\Phenikaa-Study\kì 2 2025-2026\Xử lý ngôn ngữ\VS code\Thu Data\data.csv'


if not os.path.exists(save_path):
    with open(save_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'label'])

def get_article_links(category_url, limit=600):
    links = []
    page = 1
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    # Xóa ký tự /p ở cuối nếu bạn lỡ nhập vào categories để hàm tự xử lý
    base_url = category_url.rstrip('/p').rstrip('/')
    
    while len(links) < limit:
        # KIỂM TRA: Nếu là link chuyên mục sâu (có dấu / thứ 4) thì dùng /p, ngược lại dùng -p
        # Ví dụ: /thoi-su (3 dấu /) -> -p | /kinh-doanh/quoc-te (4 dấu /) -> /p
        if base_url.count('/') >= 4:
            url = f"{base_url}/p{page}"
        else:
            url = f"{base_url}-p{page}"
            
        try:
            res = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(res.text, 'html.parser')
            items = soup.find_all(['h3', 'h2'], class_=['title-news', 'title_news'])
            
            if not items: break
            
            old_len = len(links)
            for item in items:
                a_tag = item.find('a')
                if a_tag and 'href' in a_tag.attrs:
                    link = a_tag['href']
                    if 'vnexpress.net' in link and link not in links:
                        links.append(link)
                    if len(links) >= limit: break
            
            # Nếu không tìm thêm được link mới ở trang này thì dừng
            if len(links) == old_len: break
            
            print(f"   Đang quét: {url} (Đã có {len(links)} link)", end='\r')
            page += 1
            time.sleep(0.3)
        except:
            break
    return links

def crawl_and_save_immediately(url, label_id, file_path):
    try:
        res = requests.get(url, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        title = soup.find('h1', class_='title-detail').text.strip()
        desc = soup.find('p', class_='description').text.strip() if soup.find('p', class_='description') else ""
        content = " ".join([p.text.strip() for p in soup.find_all('p', class_='Normal')])
        full_text = title + " " + desc + " " + content
        
        # LƯU NGAY LẬP TỨC VÀO FILE
        with open(file_path, 'a', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([full_text, label_id])
        return True
    except:
        return False

# Bắt đầu quy trình
print(f"--- Bắt đầu cào dữ liệu. Dữ liệu sẽ được lưu trực tiếp vào {save_path} ---")

for label_id, cat_url in categories.items():
    print(f"\nĐang lấy link cho nhãn {label_id}...")
    links = get_article_links(cat_url, limit=600)
    
    success_count = 0
    for i, link in enumerate(links):
        if crawl_and_save_immediately(link, label_id, save_path):
            success_count += 1
        
        if (i + 1) % 10 == 0:
            print(f"Nhãn {label_id}: Đã xong {i+1}/{len(links)} bài. (Đã lưu vào file)")

print(f"\n--- HOÀN THÀNH ---")
print(f"File hiện tại đang nằm ở: {os.path.abspath(save_path)}")