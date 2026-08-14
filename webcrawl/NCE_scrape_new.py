import requests
import json
from bs4 import BeautifulSoup
import time
from datetime import datetime

start_time = time.time()

f = open('urls/NCE_urls_2426.txt', 'r')
urls_raw = f.read()
urls = urls_raw.split('\n')
##
headers = {
    "User-Agent": "Mozilla/5.0 Masters Thesis Research Crawler/1.0"
}

def make_date_format(date_str):
    date_obj = datetime.strptime(date_str, "%d-%m-%Y")
    return date_obj.strftime("%d %B %Y")

def scrape_NCE_article(url):
    print(url)
    # added timeout
    response = requests.get(url, headers=headers, timeout = 30)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    try:
        title = soup.find('h1', class_='name entry-title').text
    except AttributeError:
        title = None
    
    date = url[-11:-1]
    try:
        date = make_date_format(date)
    except ValueError:
        date = None
    
    try:
        author = soup.find('a', class_='author url fn').text
    except AttributeError:
        author = None

    try:
        raw_text = soup.find('div', class_='content container').text.split('\n')
        text = []
        unwanteds = ['to receive new civil engineer\'s', 'like what you\'ve read?', 'tagged with:', 'sign in or register']
        tmp = url[-11:-1]
        url_date = tmp[-4:] + tmp[-5] + tmp[-7:-5] + tmp[-8] + tmp[:-8]
        for line in raw_text:
            if url_date in line:
                break
            elif not any(unwanted in line.lower() for unwanted in unwanteds) \
                and len(line) > 25:
                text.append(line)
    except AttributeError:
        text = None
        print('No text')

    return {
        'url': url,
        'magasine': 'New Civil Engineer',
        'title': title,
        'author': author,
        'date': date,
        'text': text
    }

#articles = [scrape_NCE_article(url) for url in urls if url]

articles = []
# modified writing loop to ensure error handling, and saving at checkpoints
for index, url in enumerate(urls):
    if not url:
        continue # to skip empty strings
    try:
        article = scrape_NCE_article(url)
        articles.append(article)
    except Exception as e:
        print(f"Error scraping {url}: {e}")
    if index % 10 == 0:
        print(f"Scraped {index} out of {len(urls)} articles")

    if index % 100 == 0: # saves every 100 articles
        with open('NCE_articles_2426.json', 'w', encoding='utf-8') as file:
            json.dump(articles, file, ensure_ascii=False, indent=4)
    

end_time = time.time()

print(f'Finished scrape, took {(end_time-start_time)/60:.2f} minutes')
##
with open('NCE_articles_2426.json', 'w', encoding='utf-8') as file:
    json.dump(articles, file, ensure_ascii=False, indent=4)