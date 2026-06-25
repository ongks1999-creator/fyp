from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import requests
import csv
import pandas as pd
import time
from http.cookies import SimpleCookie
from collections import Counter, defaultdict
import ast

def extract_additional_urls_from_url(html, url) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    parsed_url = urlparse(url)
    current_netloc = parsed_url.netloc # get domain of current url

    external_urls = []

    tags = soup.find_all("a", href = True) # searching for <a href="> tags to find external links

    for tag in tags:
        # not all tags["href"] have complete urls with domain at the front
        completed_url = urljoin(url, tag["href"]) # if href is incomplete, completes href url with domain from "url"
        # if not completed, completed_url is assigned full url from href
        domain = urlparse(completed_url).netloc

        if domain != current_netloc: # if domain is different from current domain, avoid self citation
            external_urls.append(completed_url)

    return external_urls

def test_url(cookies = None):
    """
    Used to compare the retrieved readable paragraphs compared to what is actually found on website, from manual inspection
    """
    # test for paywall for the four sources
    links = {"NCE": "https://www.newcivilengineer.com/latest/labour-and-tories-pledge-new-nuclear-deployment-in-scotland-despite-local-opposition-25-06-2024/",
             "NEI": "https://www.neimagazine.com/news/bridge-cranes-installed-at-akkuyu-2-to-enhance-operations/",
             "NUCNET": "https://www.nucnet.org/news/kyrgyzstan-6-5-2024",
             "WNN": "https://www.world-nuclear-news.org/Articles/Chinese-HTR-PM-Demo-begins-commercial-operation"}
    

    results = {}
    for source, test_url in links.items():
        # containers manually inspected and verified from webpage
        # for NCE <article> is used only once in the page, and its class text is a post specific string (non-reusable), as "article" is sufficient to search for the required nested chunks
        container = {"NEI": ["div", "main-content"], "NUCNET": ["article", "article-style"], "WNN": ["div", "article_body_wrapper"], "NCE": ["article"]}
        if source == "NUCNET":
            cookies_dict = parse_cookies(cookies) # nucnet has paywall

        else:
            cookies_dict = None

        html = extract_html(test_url, cookies = cookies_dict)
        soup = BeautifulSoup(html, "html.parser")
        if source != "NCE":  
            container = soup.find(container[source][0], class_ = container[source][1])
        else: # since None cant be used as input for class_ parameter for NCE
            container = soup.find(container[source][0])
        paragraphs = container.find_all("p")
        results[source] = len(paragraphs)

    return results

def parse_cookies(cookies: str) -> dict:
    """
    Parse the cookie string into a dictionary, for nucnet paywall
    """
    cookies_dict = {}

    if cookies:
        cookies = SimpleCookie(cookies)

        for key, morsel in cookies.items():
            cookies_dict[key] = morsel.value

    return cookies_dict

def extract_html(url, cookies = None):
    # cookies required by nucnet due to premium paywall
    # had to set user agent if not blocked by NCE
    headers = {"User-Agent": "Mozilla/5.0 Script for Research Project"}

    cookies_dict = parse_cookies(cookies)

    # error handling required in the event url is invalid
    try:
        # 5 second timeout to raise error if request is too long
        page_response = requests.get(url, timeout = 5, headers = headers, cookies = cookies_dict)
        html = page_response.text

    except Exception as e:
        print(f"Url {url} is invalid")
        return None
    
    return html

def main_loop(filepath, cookies = None):
    
    entries = []
    with open(filepath, newline = '') as csvfile: # use url and source id from paragraphs.csv
        paragraphs_csv = csv.DictReader(csvfile)
        paragraphs_csv = list(paragraphs_csv)
        # since multiple entries can be from same source url
        verified_sources = set() # keep tracked of which url has been scraped

        for index, entry in enumerate(paragraphs_csv): 
            if entry["source_id"] in verified_sources:
                continue
            verified_sources.add(entry["source_id"])
            new_csv_entry = {}
            url = entry["url"]
            new_csv_entry["source_id"] = entry["source_id"]
            new_csv_entry["url"] = url
            new_csv_entry["title"] = entry["title"]
            new_csv_entry["magazine"] = entry["magazine"]
            new_csv_entry["date"] = entry["date"]

            if entry["magazine"] == "NucNet":
                html = extract_html(url, cookies = cookies)
            else:
                html = extract_html(url)

            if html is None: # skip the failed url
                continue
            additional_urls = extract_additional_urls_from_url(html, url)
            new_csv_entry["external_urls"] = additional_urls
            entries.append(new_csv_entry)
            time.sleep(0.5) # to slow down request to the site, prevent getting blocked

            if index % 10 == 0: # progress checker
                print(f"Processed {index} out of {len(paragraphs_csv)} entries")

    df = pd.DataFrame(entries)
    df.to_csv("extracted_links.csv", index = False)

def find_threshold(external_links_per_article_dict: dict[str, Counter]) -> float:
    """
    Find the clustering of values for percentage of articles per source
    Determine a threshold value such that there is a segregation between noise and geniune links
    """
    percentages_list = []
    for source, external_links in external_links_per_article_dict.items():
        for domain, percentage in external_links.items():
            percentages_list.append(percentage) # add all percentages into a list to sort

    percentages_list.sort()

    threshold = 0
    largest_gap = 0
    for index, percentage in enumerate(percentages_list):
        if index == len(percentages_list) - 1: # prevent index error
            break
        gap = percentages_list[index + 1] - percentage # difference between current and next number
        # when the largest gap is found, the threshold is the lower value
        # at this threshold value the noises are separated from the genuine links
        if gap > largest_gap:
            largest_gap = gap
            threshold = percentage
    print(f"Threshold Value of :{threshold:.3f}%")
    
    return threshold


def filter_links(filepath) -> dict[str, set[str]]:
    article_per_source_dict = Counter() # number of articles produced by each magazine source
    external_links_per_article_dict = defaultdict(Counter) # dict where key is the magazine source, and value is a Counter, 
    # counting number of times that external link appeared for this source  

    with open(filepath, newline = '') as csvfile:
        links_csv = csv.DictReader(csvfile)
        links_csv = list(links_csv)
    
    for entry in links_csv:
        source = entry["magazine"]
        article_per_source_dict[source] += 1

        external_links = ast.literal_eval(entry["external_urls"])
        external_links_domains = set()
        for link in external_links: # use set to find the key names for the external_links_per_article_dict
            domain = urlparse(link).netloc
            external_links_domains.add(domain)

        for domain in external_links_domains:
            external_links_per_article_dict[source][domain] += 1
        
    for source, external_links in external_links_per_article_dict.items():
        total_source_articles = article_per_source_dict[source]
        for external_link in external_links:
            count = external_links[external_link]
            percentage = (count / total_source_articles) * 100
            external_links[external_link] = percentage # normalise count based on total number of articles per source

    noise_domain_links = defaultdict(set)
    threshold = find_threshold(external_links_per_article_dict)

    # creation of noise_domain_links dict
    for source, external_links in external_links_per_article_dict.items():
        for domain, percentage in external_links.items():
            if percentage > threshold:
                noise_domain_links[source].add(domain)
                #print(f"Source: {source}, External Link: {domain}, Found in {percentage:.3f}% of articles in {source}")
    
    return noise_domain_links

def edit_links_file(filepath, noise_domain_links):
    """
    Remove noise links from the original extracted_links.csv file, creates the final version
    """
    with open(filepath, newline = '') as csvfile:
        links_csv = csv.DictReader(csvfile)
        links_csv = list(links_csv)
    
    new_entries = []
    for entry in links_csv:
        new_entry = {}
        new_entry["source_id"] = entry["source_id"]
        new_entry["magazine"] = entry["magazine"]
        new_entry["date"] = entry["date"]
        new_entry["external_urls"] = []
        
        external_links = ast.literal_eval(entry["external_urls"])
        for link in external_links:
            domain = urlparse(link).netloc
            if domain not in noise_domain_links[entry["magazine"]]:
                new_entry["external_urls"].append(link)
        new_entries.append(new_entry)
    
    df = pd.DataFrame(new_entries)
    df.to_csv("filtered_extracted_links.csv", index = False)
    
    return


if __name__ == "__main__":
    
    filepath = "/Users/ongkaisheng/Desktop/ImperialCollege/FYP/fyp/paragraphs.csv"


    with open("nucnet_cookies.txt") as f: # cookies file kept on separate text file
        cookies = f.read().strip()
    #print(test_url(cookies = cookies))

    #main_loop(filepath, cookies = cookies)

    external_links_filepath = "/Users/ongkaisheng/Desktop/ImperialCollege/FYP/fyp/extracted_links.csv"
    noisy_links = filter_links(external_links_filepath)
    edit_links_file(external_links_filepath, noisy_links)
