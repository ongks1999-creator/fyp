import csv
from collections import defaultdict
import ast
import json
import pandas as pd

floor = 0.5 # change if required

def recency_scoring(paragraph_csv_file_path: str)-> dict:
    """
    Calculating recency based on how frequent articles are published for each source.
    """

    with open(paragraph_csv_file_path, newline = '') as csvfile:

        paragraphs_csv = csv.DictReader(csvfile)
        paragraphs_csv = list(paragraphs_csv)
        verified_sources = set()
        recency_dict = defaultdict(list)

        for entry in paragraphs_csv: 
            if entry["source_id"] in verified_sources: # to prevent double counting for the same article
                continue
            verified_sources.add(entry["source_id"])
            if entry["magazine"] == "Reddit":
                recency_dict[entry["magazine"]].append(int(entry["date"]) / 86400) # reddit timestamp in seconds, convert sec to days
            else:   # for other sources 
                recency_dict[entry["magazine"]].append(int(entry["date"]) / 86400000) # convert ms to days

        for key, value in recency_dict.items():
            recency_dict[key].sort() # sort dates in ascending order to find interval between articles

        for key, value in recency_dict.items():
            total_recency = 0
            for index in range(len(value)):
                if index == len(value) - 1: # avoid index error for last element
                    break
                recency = recency_dict[key][index + 1] - recency_dict[key][index]
                total_recency += recency

            average_recency = total_recency / (len(value) - 1) # divide by number of intervals
            recency_dict[key] = average_recency

    # linear scaling from floor value to 1.0
    min_recency = min(recency_dict.values())
    max_recency = max(recency_dict.values())
    for key, value in recency_dict.items():
        # the smaller the recency value the higher the recency score, hence opposite formula to provenance
        recency_dict[key] = floor + (1 - floor) * ((max_recency - value) / (max_recency - min_recency))

    return recency_dict

def provenance_scoring(filtered_extracted_links_csv_file_path: str, filtered_reddit_articles_json_file_path: str)-> dict:
    """
    Provenance calculated based on number of external citations per article.
    It is linearly scaled from 0.5 to 1.0 (0.5 is floor value)
    """
    provenance_dict = {}
    visited_source_ids = set()

    with open(filtered_extracted_links_csv_file_path, newline = '') as csvfile:
        links_csv = csv.DictReader(csvfile)
        links_csv = list(links_csv)

        for entry in links_csv:
            if entry["source_id"] in visited_source_ids: 
                continue
            visited_source_ids.add(entry["source_id"])
            provenance_dict[entry["source_id"]] = len(ast.literal_eval(entry["external_urls"])) # convert string to list

    with open(filtered_reddit_articles_json_file_path) as file:
        reddit_articles = json.load(file)
        for entry in reddit_articles:
            if entry["source_id"] in visited_source_ids: 
                continue
            visited_source_ids.add(entry["source_id"])
            # external_links for reddit articles is already in a list
            provenance_dict[entry["source_id"]] = len(entry["external_links"]) # convert string to list

    max_provenance = max(provenance_dict.values()) # max is the highest number of external citations per article
    min_provenance = min(provenance_dict.values())
    for key, value in provenance_dict.items():
        provenance_dict[key] = floor + (1 - floor) * ((value - min_provenance) / (max_provenance - min_provenance))

    return provenance_dict


def main():
    filtered_extracted_links_csv_file_path = "/Users/ongkaisheng/Desktop/ImperialCollege/FYP/fyp/filtered_extracted_links_2426.csv"
    filtered_reddit_articles_json_file_path = "/Users/ongkaisheng/Desktop/ImperialCollege/FYP/fyp/filtered_reddit_articles_2426.json"

    chunks_final_csv_file_path = "/Users/ongkaisheng/Desktop/ImperialCollege/FYP/fyp/chunks_final_2426.csv"

    #recency_dict = recency_scoring(chunks_final_csv_file_path)
    # freezing recency scores for 24/26 data
    recency_dict = {"Nuclear Engineering International": 1.0, "NucNet": 0.9060278201577280, "Reddit": 0.8839794498904370, "World Nuclear News": 0.7499772938021270, "New Civil Engineer": 0.5}
    provenance_dict = provenance_scoring(filtered_extracted_links_csv_file_path, filtered_reddit_articles_json_file_path)
    # iterate through the final chunk list that contain all source ids
    with open(chunks_final_csv_file_path, newline = '') as csvfile:
        entries_csv = csv.DictReader(csvfile)
        entries = list(entries_csv)

    sources = []
    visited_source_ids = set()
    for entry in entries:     
        if entry["source_id"] in visited_source_ids: 
            continue
        source_entry = {}
        visited_source_ids.add(entry["source_id"])
        source_entry["source_id"] = entry["source_id"]
        source_entry["url"] = entry["url"]
        source_entry["title"] = entry["title"]
        source_entry["magazine"] = entry["magazine"]
        source_entry["date"] = entry["date"]
        source_entry["recency_score"] = recency_dict[entry["magazine"]]
        source_entry["provenance_score"] = provenance_dict[entry["source_id"]]
        # tentative source credibility score, fixed score
        # highest was given to NEI as it is a trade publication
        # WNN and nucnet both report nuclear news but WNN is accreditted by World nuclear association
        # NCE has the second lowest score as it is a general civil engineering publication
        # reddit lowest as it is a social news platform
        if entry["magazine"] == "Reddit":
            source_entry["source_credibility"] = 0.5

        elif entry["magazine"] == "Nuclear Engineering International":
            source_entry["source_credibility"] = 0.9

        elif entry["magazine"] == "World Nuclear News":
            source_entry["source_credibility"] = 0.8

        elif entry["magazine"] == "NucNet":
            source_entry["source_credibility"] = 0.7

        elif entry["magazine"] == "New Civil Engineer":
            source_entry["source_credibility"] = 0.6


        sources.append(source_entry)
    
    sources_df = pd.DataFrame(sources)
    pd.DataFrame.to_csv(sources_df, "sources_2426.csv", index = False)



if __name__ == "__main__":
    main()
