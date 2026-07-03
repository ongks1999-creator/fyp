import csv
from collections import defaultdict
import ast

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
            # convert millisecond date to days
            recency_dict[entry["magazine"]].append(int(entry["date"]) / 86400000) # appending dates of all articles for each source


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

def provenance_scoring(filtered_extracted_links_csv_file_path: str)-> dict:
    """
    Provenance calculated based on number of external citations per article.
    It is linearly scaled from 0.5 to 1.0 (0.5 is floor value)
    """

    with open(filtered_extracted_links_csv_file_path, newline = '') as csvfile:
        links_csv = csv.DictReader(csvfile)
        links_csv = list(links_csv)
        provenance_dict = {}
        visited_source_ids = set()
        for entry in links_csv:
            if entry["source_id"] in visited_source_ids: 
                continue
            visited_source_ids.add(entry["source_id"])
            provenance_dict[entry["source_id"]] = len(ast.literal_eval(entry["external_urls"])) # convert string to list

        max_provenance = max(provenance_dict.values()) # max is the highest number of external citations per article
        min_provenance = min(provenance_dict.values())
        for key, value in provenance_dict.items():
            provenance_dict[key] = floor + (1 - floor) * ((value - min_provenance) / (max_provenance - min_provenance))

    return provenance_dict
