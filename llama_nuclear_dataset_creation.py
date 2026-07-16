import csv
import requests
import json
import random
from collections import defaultdict
import pandas as pd

def read_chunk_csv(file_path: str):

    with open(file_path, newline = '') as csvfile:
        chunks_csv = csv.DictReader(csvfile)
        chunks_csv = list(chunks_csv) # convert dic to list of dicts
        
    return chunks_csv

def call_llama_api(prompt: str):
    """
    Encountered decode error, included error handling
    """

    while True:

        response = requests.post("http://localhost:11434/api/generate", json = {"model": "llama3.2", "prompt": prompt, "stream": False})

        result = response.json() # convert llama string output to dict

        try:
            output = json.loads(result["response"]) # value of response is in string format, convert to dict

            return output

        except json.JSONDecodeError as e:
            print(f"Error encountered: {e}.")

def claim_extraction_from_chunk(chunk: list):

    chunk_text = chunk["chunk"]
    prompt = f"Extract an atomic claim from the following text: {chunk_text}. From this atomic claim, extract the company name. Return a JSON with fields: \"claim\" and \"company\". If there is no company mentioned in the claim, return a None under \"company\""

    claim_output = call_llama_api(prompt)

    return claim_output # dict containing claim and company name

def add_support_entry(chunk: list):
    """
    extracts a claim from that chunk and add as a support entry
    """
    claim_output = claim_extraction_from_chunk(chunk)

    return {"source_id": chunk["source_id"], "magazine": chunk["magazine"], "company": claim_output["company"], "claim": claim_output["claim"], "chunk": chunk["chunk"], "label": "SUPPORT"}

def add_contradict_entry(claim_output: dict, chunks: list):
    """
    For that extracted claim, iterate through the rest of the chunks to find one that refutes it
    """
    for chunk in chunks:
        claim_text = claim_output["claim"]
        chunk_text = chunk["chunk"]
        prompt = f"Given the claim:{claim_text}, does this text: {chunk_text} support or refute the claim. Return a JSON with field: \"label\" with value either \"SUPPORT\" or \"CONTRADICT\""

        output = call_llama_api(prompt)

        if output["label"] == "CONTRADICT":

            return {"source_id": chunk["source_id"], "magazine": chunk["magazine"], "company": claim_output["company"], "claim": claim_output["claim"], "chunk": chunk["chunk"], "label": "CONTRADICT"}

def add_NEI_entry(claim_output: dict, chunks:list):
    """
    for that extracted claim, we randomly select a chunk and used LLaMA to verify it there is sufficient information to verify that claim
    """
    label = None

    while label != "Not-Enough-Information":

        chunk = random.choice(chunks) # randomly pick a chunk from the chunks list and pair it with that input claim
        claim_text = claim_output["claim"]
        chunk_text = chunk["chunk"]
        prompt = f"Given the claim:{claim_text}, does this text: {chunk_text} have enough information to verify the claim. Return a JSON with field: \"label\" with value either \"Not-Enough-Information\" or \"Enough-Information\""

        output = call_llama_api(prompt) 
        label = output["label"]

    return {"source_id": chunk["source_id"], "magazine": chunk["magazine"], "company": claim_output["company"], "claim": claim_output["claim"], "chunk": chunk["chunk"], "label": "NEI"}

def sample_chunks(chunks: list)-> dict:
    """
    Randomly sample 25 chunk entries from each of the four magazines
    Reddit is not used as the nature of the chunk text is informal and is not appropriate to create a 'gold standard' dataset
    """

    magazine_chunks = defaultdict(list)
    for chunk in chunks: # group chunks by magazines
        magazine_chunks[chunk["magazine"]].append(chunk)

    sampled_magazine_chunks = defaultdict(list)

    for magazine, chunks in magazine_chunks.items():
        random.shuffle(chunks)
        sampled_chunks = random.sample(chunks, 25) # sample 25 chunks from each magazine
        sampled_magazine_chunks[magazine] = sampled_chunks

    return sampled_magazine_chunks

def create_dataset(sampled_magazine_chunks: dict, chunks: list):
    """
    Using sampled magazine chunks to create the dataset, this ensures the 100 claim entries are equally sampled entries from the four sources
    "chunks" is from the entire dataset, and is only used when the evidence is used to create contradict or NEI entries
    To ensure the 100 entries contain 25 sampled chunks for each magazine,
    9 are used to create SUPPORT entries, 8 for CONTRADICT entries, 8 for NEI
    """
    dataset = []

    for sampled_chunks in sampled_magazine_chunks.values():
        for i in range(25): 
            if i < 9:
                dataset.append(add_support_entry(sampled_chunks[i]))

            elif i < 17:
                claim_output = claim_extraction_from_chunk(sampled_chunks[i])
                dataset.append(add_contradict_entry(claim_output, chunks))

            else:
                claim_output = claim_extraction_from_chunk(sampled_chunks[i])
                dataset.append(add_NEI_entry(claim_output, chunks))

    df = pd.DataFrame(dataset)
    df.to_csv("finetune_dataset.csv", index = False)

if __name__ == "__main__":
    file_path = "/root/fyp/chunks.csv" # update
    chunks_csv = read_chunk_csv(file_path)
    sampled_magazine_chunks = sample_chunks(chunks_csv)
    create_dataset(sampled_magazine_chunks, chunks_csv)