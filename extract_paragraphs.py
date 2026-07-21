from chunk_index import url_normalisation, create_source_id
from transformers import AutoTokenizer
import pandas as pd
import json

# Since we using Scibert model, we use the same model for our tokenizer
tokenizer = AutoTokenizer.from_pretrained("/homes/ko25/Desktop/fyp/scibert_finetuned_model")

def extract_paragraphs_from_article(article: dict) -> list[dict]:
    """
    Takes each article in the form of a dictionary and create a list of dictionaries
    Each dictionary represents each paragraph within that article

    This function is used for paragraph.csv creation
    """
    out = []
    source_id = create_source_id(article["url"]) # kept outside to prevent repeated calls of the two functions
    url = url_normalisation(article["url"])
    for index, paragraph in enumerate(article["text"]):
        dic = {}
        dic["source_id"] = source_id
        dic["url"] = url
        dic["title"] = article["title"]
        dic["magazine"] = article["magasine"]
        dic["date"] = article["date"]
        dic["paragraph_index"] = index
        dic["text"] = paragraph
        dic["token_count"] = len(tokenizer.tokenize(paragraph))
        out.append(dic)

    return out

def entry_check(articles_filepath) -> list[dict]:
    """
    Check if the article has all fields with non-empty text
    """
    with open(articles_filepath) as f:
        articles = json.load(f) # articles is a list of dict
    
    invalid_articles_list = []
    for article in articles:
        if not all([article["url"], article["title"], article["magasine"], article["date"], article["text"]]):
            invalid_articles_list.append(article)
            
    return invalid_articles_list

def create_paragraphs_csv(articles_filepath) -> None:
    """
    Create the paragraphs.csv file by converting json file into list of dicts then into df
    """
    with open(articles_filepath) as f:
        articles = json.load(f) 
    
    dataframe_list = []
    for article in articles:
        paragraphs = extract_paragraphs_from_article(article) # convert each dict into a list of dicts
        dataframe_list.extend(paragraphs) # prevent adding the list of dict as an element
    
    df = pd.DataFrame(dataframe_list)
    #df.to_csv("paragraphs.csv", index = False) # write dataframe to csv
    df.to_csv("paragraphs_extra.csv", index = False)
    return

if __name__ == "__main__":
    file_path = "/homes/ko25/Desktop/fyp/filtered_reddit_articles.json" #"/Users/ongkaisheng/Desktop/ImperialCollege/FYP/finalcode/articles/filtered_articles.json"
    invalid_entries = entry_check(file_path)
    print(invalid_entries)
    print([invalid["magasine"] for invalid in invalid_entries])
    print(f"Number of invalid entries: {len(invalid_entries)}")
    create_paragraphs_csv(file_path)

