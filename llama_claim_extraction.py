import csv
from llama_nuclear_dataset_creation import call_llama_api
from RAG_uncertainty_label import uncertainty_label
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
from chunk_index import create_source_id

def read_chunk_csv(file_path: str):

    with open(file_path, newline = '') as csvfile:
        chunks_csv = csv.DictReader(csvfile)
        chunks_csv = list(chunks_csv) # convert dic to list of dicts
        
    return chunks_csv

def claim_extraction_from_chunk(chunk: list):

    chunk_text = chunk["chunk"]
    # refined prompt to ensure nuclear claims are complete, and company name if absent given a null for json format
    prompt = (f"Extract an atomic claim relating to the maturity, deployment, or technicalities of Small Modular Reactors (SMRs) from the following text: {chunk_text}."
    f"If there is no information on SMRs available, extract information relating to the nuclear industry. The claim MUST be complete and verifiable."
    f"Claim must be less than a 100 tokens. From this atomic claim, extract all company names and return them as a list of strings."
    f"Return ONLY a JSON object with ONLY these two keys: \"claim\" and \"company\". If there are no company names mentioned in the claim, return a null value under \"company\""
    f"For example, {{\"claim\": \"ABC plans to build two new SMR reactors\", \"company\": [\"ABC\"]}}"
    f"Another example,{{\"claim\": \"ABC and XYZ are working together to develop five new SMR reactors by 2024\", \"company\": [\"ABC\", \"XYZ\"]}}")
    
    counter_limit = 10
    counter = 0
    while True:
        counter += 1
        if counter > counter_limit:
            return claim_output # if limit reached, take the final attempt
        claim_output = call_llama_api(prompt)
        # ensures the key names are exactly what we want
        if "claim" in claim_output and "company" in claim_output:
                break
    return claim_output # dict containing claim and company names


if __name__ == "__main__":
    chunks_final_file_path = "/vol/bitbucket/ko25/fyp/chunks_final_2426.csv"
    chunks = read_chunk_csv(chunks_final_file_path)

    articles_filepath = "/vol/bitbucket/ko25/fyp/webcrawl/filtered_articles_2426.json"
    # keep only chunks from smr related articles, only extract claims from smr related chunks
    with open(articles_filepath, "r") as file:
        articles = json.load(file)

    smr_articles = set()
    for article in articles:
        # combine title and body paragraph text into a single string to search
        overall_text = (article.get("title") or "").lower() + " " + " ".join(article.get("text") or []).lower()

        if "small modular reactor" in overall_text or "smr" in overall_text:
            smr_articles.add(create_source_id(article["url"]))

    chunks_2426 = [] # for 24/26 data, we extract claims from chunks from smr related articles
    for chunk in chunks:
        if chunk["source_id"] in smr_articles or chunk["magazine"] == "Reddit": # all reddit chunks already filtered to be smr related
            chunks_2426.append(chunk)
    print(f"Total number of chunks after filtering is: {len(chunks_2426)} out of initial total chunks of: {len(chunks)}")
    # Initialise scibert model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # to update again
    scibert_tokenizer = AutoTokenizer.from_pretrained("/vol/bitbucket/ko25/fyp/scibert_finetuned_model")
    scibert_model = AutoModelForSequenceClassification.from_pretrained("/vol/bitbucket/ko25/fyp/scibert_finetuned_model")
    scibert_model.to(device)
    threshold = 0.3
    claim_token_limit = 100

    claims_list = []
    for index, chunk in enumerate(chunks_2426):
        claim = claim_extraction_from_chunk(chunk)
        # faced keyerror when claim generated was a null value, use get method to check if key exist
        if not isinstance(claim.get("claim"), str):
            continue # skip for that claim
        if index % 100 == 0:
            print(f"Processing chunk {index} out of {len(chunks_2426)}")
            
        token_count = len(scibert_tokenizer.tokenize(claim["claim"]))

        if token_count > claim_token_limit:
            continue # only keep claims that are under 100 tokens
        stance_score, label = uncertainty_label(claim["claim"], chunk["chunk"], scibert_tokenizer, scibert_model, threshold, device)

        if stance_score > threshold: # keep claim if stance above threshold val
            claim_entry = {}
            claim_entry["claim_id"] = chunk["chunk_id"] + "_claim"
            claim_entry["chunk_id"] = chunk["chunk_id"]
            claim_entry["url"] = chunk["url"]
            claim_entry["title"] = chunk["title"]
            claim_entry["magazine"] = chunk["magazine"]
            claim_entry["date"] = chunk["date"]
            claim_entry["claim"] = claim["claim"]
            claim_entry["company"] = claim["company"]
            claim_entry["token_count"] = token_count
            claims_list.append(claim_entry)

            if len(claims_list) % 100 == 0: # checkpoint saved every 100 claims
                claims_df = pd.DataFrame(claims_list)
                pd.DataFrame.to_csv(claims_df, "claims_2426.csv", index = False)
                print(f"Checkpoint saved at chunk {index} / {len(chunks_2426)}, current claim count: {len(claims_list)}")

    claims_df = pd.DataFrame(claims_list)
    pd.DataFrame.to_csv(claims_df, "claims_2426.csv", index = False)
