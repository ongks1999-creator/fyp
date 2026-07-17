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

        response = requests.post("http://localhost:11434/api/generate", json = {"model": "llama3.1:8b", "prompt": prompt, "stream": False, "format": "json"})

        result = response.json() # convert llama string output to dict

        try:
            output = json.loads(result["response"]) # value of response is in string format, convert to dict
            # added to prevent keyerror due to case sensitivity, encountered for "company"
            output = {key.lower(): value for key, value in output.items()} # standardise keys to lower case
            return output

        except json.JSONDecodeError as e:
            print(f"Error encountered: {e}.")
            print(f"raw results: {result}")

def claim_extraction_from_chunk(chunk: list):

    chunk_text = chunk["chunk"]
    # refined prompt to ensure nuclear claims are complete, and company name given a null for json format
    prompt = f"Extract an atomic claim relating to the maturity, deployment, or technicalities of Small Modular Reactors (SMRs) from the following text: {chunk_text}. If there is no information on SMRs available, extract information relating to the nuclear industry. The claim MUST be complete and verifiable. From this atomic claim, extract the company name. Return ONLY a JSON object with ONLY these two keys: \"claim\" and \"company\". If there is no company mentioned in the claim, return a null value under \"company\" For example, {{\"claim\": \"ABC plans to build two new SMR reactors\", \"company\": \"ABC\"}}"
    
    retry_count = 0

    while True:
        retry_count += 1
        claim_output = call_llama_api(prompt)
        # ensures the key names are exactly what we want
        if "claim" in claim_output and "company" in claim_output:
            if retry_count > 10:
                print("retry limit reached, taking best attempt")
                break # to prevent infinite loop, as it gets stuck on a particular claim, we take the best attempt after 10 retries
            # validation step, by calling llama again to verify its output before adding that entry
            claim_text = claim_output["claim"]
            validation_prompt = f"Validate the following claim: {claim_text} and verify if it is related to the nuclear domain and is supported by the text: {chunk_text}. Return ONLY a JSON object with exactly this key: \"validity\" with exactly the value \"yes\" or \"no\""
            validation_output = call_llama_api(validation_prompt)
            if validation_output.get("validity") == "yes": # to prevent llama from returning a key name that is different from "validity" then getting keyerror
                break

    return claim_output # dict containing claim and company name

def add_support_entry(chunk: list):
    """
    extracts a claim from that chunk and add as a support entry
    """
    claim_output = claim_extraction_from_chunk(chunk)

    return {"claim": claim_output["claim"], "company": claim_output["company"],"chunk": chunk["chunk"], "label": "SUPPORT"}

def add_contradict_entry(chunk: dict):
    """
    Using the chunk provided, we generate a claim that directly refutes the chunk
    """
    chunk_text = chunk["chunk"]
    prompt = f"Given this text: {chunk_text}, if it contains information relating to the maturity, deployment, or technicalities of Small Modular Reactors (SMRs),generate an atomic claim that directly REFUTES the text. If the text does not contain any information on SMRs, generate a claim relating to the nuclear industry that directly refutes the text. The claim MUST be complete and verifiable. Contradiction can be defined as either negation of a specific fact in the text or a modification of the specific fact within the text. From this atomic claim, extract the company name. Return ONLY a JSON object with ONLY these two keys: \"claim\" and \"company\". If there is no company mentioned in the claim, return a null value under \"company\" For example, the original text could be \"ABC plans to build 5 SMRs\" with the generated claim being {{\"claim\": \"ABC plans to build two new SMR reactors\", \"company\": \"ABC\"}}"
    retry_count = 0
    while True:
        retry_count += 1
        claim_output = call_llama_api(prompt)
        
        if "claim" in claim_output and "company" in claim_output:
            if retry_count > 10:
                print("retry limit reached, taking best attempt")
                break # take best attempt to prevent infinite loop
            claim_text = claim_output["claim"]
            validation_prompt = f"Does the following claim: {claim_text} contradict the text: {chunk_text} in terms of negation of a fact or a modification of certain details in the text? Return ONLY a JSON object with exactly this key: \"validity\" with exactly the value \"yes\" or \"no\""
            validation_output = call_llama_api(validation_prompt)
            if validation_output.get("validity") == "yes":
                break


    return {"claim": claim_output["claim"], "company": claim_output["company"], "chunk": chunk["chunk"], "label": "CONTRADICT"}

def add_NEI_entry(claim_output: dict, chunks:list):
    """
    for that extracted claim, we randomly select a chunk and used LLaMA to verify it there is sufficient information to verify that claim
    """
    label = None
    retry_count = 0
    while label != "Not-Enough-Information":

        chunk = random.choice(chunks) # randomly pick a chunk from the chunks list and pair it with that input claim
        claim_text = claim_output["claim"]
        chunk_text = chunk["chunk"]
        prompt = f"Given the claim:{claim_text}, does this text: {chunk_text} have enough information to verify the claim. Return ONLY a JSON object with exactly this key: \"label\" with value either \"Not-Enough-Information\" or \"Enough-Information\""
        retry_count += 1
        output = call_llama_api(prompt) 
        label = output["label"]
        # additional validation check
        if label == "Not-Enough-Information":
            if retry_count > 10:
                print("retry limit reached, taking best attempt")
                break # take best attempt after 10 retries
            validation_prompt = f"Validate the following claim: {claim_text} and verify if it is related to the nuclear domain or on Small Modular Reactors, and is NOT VERIFIABLE by the text: {chunk_text}. Return ONLY a JSON object with exactly this key: \"validity\" with exactly the value \"yes\" or \"no\""
            validation_output = call_llama_api(validation_prompt)
            if validation_output.get("validity") == "yes":
                break
            else:
                label = None # reset label to stay within loop, since that entry is invalid

    return {"claim": claim_output["claim"], "company": claim_output["company"], "chunk": chunk["chunk"], "label": "NEI"}

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
                print("added support entry")

            elif i < 17:
                dataset.append(add_contradict_entry(sampled_chunks[i]))
                print("added contradict entry")

            else:
                claim_output = claim_extraction_from_chunk(sampled_chunks[i])
                dataset.append(add_NEI_entry(claim_output, chunks))
                print("added NEI entry")

    df = pd.DataFrame(dataset)
    df.to_csv("finetune_dataset.csv", index = False)

if __name__ == "__main__":
    file_path = "/root/fyp/chunks.csv" # update
    chunks_csv = read_chunk_csv(file_path)
    sampled_magazine_chunks = sample_chunks(chunks_csv)
    create_dataset(sampled_magazine_chunks, chunks_csv)