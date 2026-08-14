# WNN print test
import requests
import json
from bs4 import BeautifulSoup
import re
from datetime import datetime
f = open('urls/WNN_urls.txt', 'r')
urls_raw = f.read()
urls = urls_raw.split('\n')

headers = {"User-Agent": "Mozilla/5.0 Masters Thesis Research Crawler/1.0"}



def scrape_WNN_article(url):
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    ## changed from articlebody to articlebodywrapper
    article_all = soup.find('div', class_= 'article_body_wrapper').get_text(separator="\n", strip=True)
    text_content = article_all.split('\n')
    title = text_content[0]
    try: # prevent crashing if date not retrieved
        date = datetime.strptime(re.search(r'"datePublished"\s*:\s*"([^"]+)"', response.text).group(1)[:10], "%Y-%m-%d").strftime("%d %B %Y") # extract and convert time into needed format
    except AttributeError:
        date = None
    #if len(date) > len('01 September 2024'):
    #    date = soup.find('div', class_='col-md-8 ArticleBody').find_all('p')[1].text


    unwanteds = ['related topics']
    text=[]
    if text_content:
        for line in text_content:
            if not any(unwanted in line.lower() \
                    for unwanted in unwanteds) \
                        and len(line) > 25 and line not in title:
                text.append(line)
    return {
        'url': url,
        'magasine': 'World Nuclear News',
        'title': title,
        'author': None,
        'date': date,
        'text': text
    }

articles = []
for url in urls:
    if url:
        articles.append(scrape_WNN_article(url))
        print(url)

#articles = [scrape_WNN_article(url) for url in urls if url]

with open('WNN_articles_2426.json', 'w', encoding='utf-8') as file:
    json.dump(articles, file, ensure_ascii=False, indent=4)