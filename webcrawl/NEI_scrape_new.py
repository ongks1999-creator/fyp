# NEI print test
import requests
import json
from bs4 import BeautifulSoup
import time
import re
from datetime import datetime

start_time = time.time()

f = open('urls/NEI_urls_2426.txt', 'r')
urls_raw = f.read()
urls = urls_raw.split('\n')

headers = {
    "User-Agent": "Mozilla/5.0 Masters Thesis Research Crawler/1.0"
}

def scrape_NEI_article(url):
    print(url)
    # time out to prevent hanging
    response = requests.get(url, headers=headers, timeout = 30)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    try: # no more article header title
        title = soup.find('h1').get_text()
    except AttributeError:
        title = None
    
    try:
        # regex pattern skipping any whitespaces before after semi colon, and capturing values from starting quote to ending quote
        date = datetime.strptime(re.search(r'"datePublished"\s*:\s*"([^"]+)"', response.text).group(1)[:10], "%Y-%m-%d").strftime("%d %B %Y") # extract and convert time into needed format
    except AttributeError:
        date = None
    
    try:
        author = re.search(r'"author"\s*:\s*"([^"]+)"', response.text).group(1)
    except AttributeError:
        author = None

    try:
        header_text = soup.find('p', class_='article-header__excerpt').get_text()
    except AttributeError:
        header_text = ''
        
    try:
        article_all = header_text + soup.find('section', class_= 'article-content').get_text(separator="\n", strip=True)
        text_content = article_all.split('\n')
        unwanteds = ["share this article", "sign up", "image courtesy of", "partner content", "copy link", "share on", 'give your business an edge']
        text = []
        if text_content:
            for line in text_content:
                if not any(unwanted in line.lower() \
                    for unwanted in unwanteds) \
                            and len(line) > 25:
                    text.append(line)
    except AttributeError:
        text = None
    return {
        'url': url,
        'magasine': 'Nuclear Engineering International',
        'title': title,
        'author': author,
        'date': date,
        'text': text
    }

#articles = [scrape_NEI_article(url) for url in urls if url] # no need to skip first 2 entries

articles = []
# modified writing loop to ensure error handling, and saving at checkpoints
for index, url in enumerate(urls):
    if not url:
        continue # to skip empty strings
    try:
        article = scrape_NEI_article(url)
        articles.append(article)
    except Exception as e:
        print(f"Error scraping {url}: {e}")
    if index % 10 == 0:
        print(f"Scraped {index} out of {len(urls)} articles")

    if index % 100 == 0: # saves every 100 articles
        with open('NEI_articles_2426.json', 'w', encoding='utf-8') as file:
            json.dump(articles, file, ensure_ascii=False, indent=4)
    

end_time = time.time()

print(f'Finished scrape, took {end_time-start_time:.2f} seconds')

with open('NEI_articles_2426.json', 'w', encoding='utf-8') as file: # final save
        json.dump(articles, file, ensure_ascii=False, indent=4)
    
