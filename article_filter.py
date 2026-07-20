import json
from chunk_index import url_normalisation, create_source_id

def read_json_file(filepath: str) -> list:

    with open(filepath, "r") as file:
        # jsonl file has one json object per line
        data = [json.loads(entry) for entry in file]


    return data

def filter_articles(post_data, comments_data):
    """
    Filter articles that contain "small modular reactor" or "smr" in title or body text
    If so, add the comments under that post to the overall text of the article
    We do not add comments that are deleted or removed
    """
    filtered_data = []
    for entry in post_data:
        if "small modular reactor" in entry["title"].lower() or "small modular reactor" in entry["selftext"].lower() or "smr" in entry["title"].lower() or "smr" in entry["selftext"].lower():
            if entry["selftext"] == "[deleted]" or entry["selftext"] == "[removed]":
                continue
            new_entry = {}
            url = "https://reddit.com" + entry["permalink"] # since permalink does not have the reddit.com
            new_entry["source_id"] = create_source_id(url)
            new_entry["url"] = url_normalisation(url)
            new_entry["title"] = entry["title"]
            new_entry["magasine"] = "Reddit" # standardise to "magasine" for my paragraph extraction pipeline 
            new_entry["date"] = entry["created_utc"]
            if entry["selftext"]: # if post has text, then is added to overall text
                text = entry["selftext"].split("\n\n") # split based on reddits double newline, to shrink paragraph size
                text = [paragraph.strip() for paragraph in text] # remove whitespaces
            else:
                text = []

            for comment in comments_data:
                if comment["link_id"][3:] == entry["id"]:
                    if comment["body"] == "[deleted]" or comment["body"] == "[removed]":
                        continue
                    comment_text = comment["body"].split("\n\n")
                    comment_text = [paragraph.strip() for paragraph in comment_text]
                    text.extend(comment_text) # list of strings, for standardisation for paragraph extraction code

            new_entry["text"] = text

            filtered_data.append(new_entry)


    return filtered_data


def main():
    post_filepaths = ["/vol/bitbucket/ko25/reddit/r_energy_posts.jsonl", "/vol/bitbucket/ko25/reddit/r_nuclear_posts.jsonl", "/vol/bitbucket/ko25/reddit/r_NuclearPower_posts.jsonl"]
    comments_filepaths = ["/vol/bitbucket/ko25/reddit/r_energy_comments.jsonl", "/vol/bitbucket/ko25/reddit/r_nuclear_comments.jsonl", "/vol/bitbucket/ko25/reddit/r_NuclearPower_comments.jsonl"]
    filtered_articles = []
    for post_filepath, comments_filepath in zip(post_filepaths, comments_filepaths):
        post_data = read_json_file(post_filepath)
        comments_data = read_json_file(comments_filepath)
        filtered_articles.extend(filter_articles(post_data, comments_data))

    with open("filtered_reddit_articles.json", "w") as file:
        json.dump(filtered_articles, file, indent = 1)

if __name__ == "__main__":
    main()

