import re
import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

headers = {
    "User-Agent": "Mozilla/5.0 Masters Thesis Research Crawler/1.0"
}

os.makedirs("urls", exist_ok = True) # create folder for new 24/26 urls
lower_limit_date = "2024-06-15"

def extract_url_from_sitemap(sitemap):
    """
    from site map, we extract a list of tuples, each tuple contains the url link and its date modified
    date modified is to check for time window
    use regex pattern non greedy search to extract the content between <loc> and between <lastmod>
    """
    xml = requests.get(sitemap, headers = headers, timeout = 30).text

    url_list = re.findall(r"<loc>(.*?)</loc>\s*<lastmod>(.*?)</lastmod>", xml)

    return url_list


def create_url_file(url_list, magazine_name):
    """
    Takes in list of url strings, and return txt file with url written at each line
    """
    url_text = "\n".join(url_list)

    with open(f"urls/{magazine_name}_urls_2426.txt", "w") as f:
        f.write(url_text)

    print(f"Created {magazine_name}_urls_2426.txt with {len(url_list)} urls")


def collect_WNN_urls():
    url_list = extract_url_from_sitemap("https://www.world-nuclear-news.org/sitemap.xml")
    collected_urls = []
    for url, date in url_list:
        # select url of articles that are within time range
        if date >= lower_limit_date and "/articles/" in url: # date is in ISO format
            collected_urls.append(url)

    create_url_file(collected_urls, "WNN")

def collect_NEI_NCE_urls():
    """
    For NEI and NCE their sitemap have sitemaps urls within it, need apply extraction function twice
    """
    NCE_sitemap_urls = extract_url_from_sitemap("https://www.newcivilengineer.com/sitemap.xml")
    NEI_sitemap_urls = extract_url_from_sitemap("https://www.neimagazine.com/sitemap.xml")
    NCE_collected_urls = []
    NEI_collected_urls = []
    for sitemap_url, date in NCE_sitemap_urls:
        if date >= lower_limit_date and "post-sitemap" in sitemap_url: # check for sitemap that contains the relevant articles
            url_list = extract_url_from_sitemap(sitemap_url)
            for url, url_date in url_list:
                if url_date >= lower_limit_date and url.count("/") > 3: # those url links with more than 3 "/" are valid
                    NCE_collected_urls.append(url)
            time.sleep(0.5) # need to have delays, due to repeated request to the sitemap server
                
    for sitemap_url, date in NEI_sitemap_urls:
        if date >= lower_limit_date and "post-sitemap" in sitemap_url: # same check as NCE
            url_list = extract_url_from_sitemap(sitemap_url)
            for url, url_date in url_list:
                if url_date >= lower_limit_date and "/news/" in url: # specific notation for nei articles as used by tiril
                    NEI_collected_urls.append(url)
            time.sleep(0.5)


    create_url_file(NCE_collected_urls, "NCE")
    create_url_file(NEI_collected_urls, "NEI")

def collect_NucNet_urls():
    """
    NucNet has no sitemap, collecting urls from news page listing
    """
    collected_urls = []
    invalid_page = 0
    for page in range(1, 300):
        html = requests.get(f"https://www.nucnet.org/news?page={page}", headers = headers, timeout = 30).text
        soup = BeautifulSoup(html, "html.parser")

        urls = [] # urls belonging to articles collected for that page

        tags = soup.find_all("a", href = True)
        date_tags_list = soup.find_all("time") # searching for <time> tags
        date_tags = []
        for date_tag in date_tags_list:
            date_tags.append(date_tag.get("datetime", ""))

        if all(date < lower_limit_date for date in date_tags): # check if this page has articles out of date range
            invalid_page += 1
            if invalid_page >= 5: # incase there are a few consecutive invalid pages but subsequent there are valid page
                break # no more valid pages, at the end of search
        else:
            invalid_page = 0 # reset count, that means we found a valid page and continue checking

        for tag in tags:
            # not all tags["href"] have complete urls with domain at the front
            completed_url = urljoin("https://www.nucnet.org", tag["href"])
            if "/news/" in completed_url and completed_url[-4:].isdigit(): # a few href return with /news/ but actual article have date numbers at back
                urls.append(completed_url) # need filter out non article urls as need to match no. of dates to no. of articles
                # only articles have date tags
        urls_remove_dup = {} # same article may be repeated, which causes mismatch with timetags as there are a few urls for the same article but only one timetag
        for url in urls:
            urls_remove_dup[url] = 1
        for (url, date) in zip(urls_remove_dup.keys(), date_tags):
            if date >= lower_limit_date:
                collected_urls.append(url)


        time.sleep(0.5)

    create_url_file(collected_urls, "NucNet")

if __name__ == "__main__":
    collect_WNN_urls()
    collect_NEI_NCE_urls()
    collect_NucNet_urls()