import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
import pandas as pd


def embed_csv_files(filepath: str, data_type: str)-> list:
    """
    Loads our chunk or claim csv file and embeds the text content
    Returns both the embedded chunks/claims, original chunks/claims and their IDs in list form
    """
    # load csv file
    with open(filepath, newline = '') as data_file:
        data = csv.DictReader(data_file)
        data = list(data) # to allow reiteration
        if data_type == "chunks": # chunks.csv
            content = [entry["chunk"] for entry in data]
            ids = [entry["chunk_id"] for entry in data]
        else: # claims.csv
            content = [entry["claim"] for entry in data]
            ids = [entry["claim_id"] for entry in data]

    # we embed the text using sentence transformers library, pooling is done automatically
    model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    model = SentenceTransformer(model_name)
    embeddings = model.encode(content)

    return embeddings, content, ids

def build_FAISS_index(embeddings: np.ndarray):
    """
    Build the FAISS index from the embedded chunks for semantic search
    """
    # FAISS index
    dimension = embeddings.shape[1] # embedding is np array of shape (num of chunk entries, number of dimension per vector)
    FAISS_index = faiss.IndexFlatL2(dimension) # L2 norm distance in dimensional space
    FAISS_index.add(embeddings) # add embeddings into the FAISS index, each data point projected onto dimensional space

    return FAISS_index


def uncertainty_label(claim: str, relevant_chunk: str, scibert_tokenizer, scibert_model, threshold, device):
    """
    Feed claim + relevant chunk into pre-trained scibert to get stance score and label
    """
    encodings = scibert_tokenizer(claim, relevant_chunk, padding = True, truncation = True, return_tensors = "pt")

    # no batch labels for our inference case
    # shift data to same device as model
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)
    token_type_ids = encodings["token_type_ids"].to(device)

    with torch.no_grad(): # no need for grad during inference
        output = scibert_model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
    probabilities = torch.softmax(output.logits, dim = -1)  # use softmax activation to get probabilities for each label
    # probabilities is 2d tensor containing only one entry
    P_NEI, P_CONTRADICT, P_SUPPORT = probabilities[0][0].item(), probabilities[0][1].item(), probabilities[0][2].item()
    P_STANCE = P_SUPPORT - P_CONTRADICT # stance score definition
                
    if P_STANCE > threshold: # if stance score is above the threshold, the label is SUPPORT
        label = "SUPPORT"

    elif P_STANCE < -threshold: # if less than neg threshold is CONTRADICT
        label = "CONTRADICT"

    else:
        label = "NEI"  # else NEI labels

    return P_STANCE, label

def main():
    top_N = 5 # top N relevant chunks
    threshold = 0.3 # based on what we finetuned
    claim_file_path = "/homes/ko25/Desktop/fyp/claims.csv"
    chunk_file_path = "/homes/ko25/Desktop/fyp/chunks.csv"
    # embed both chunk and claim csv
    claim_embeddings, claims, claim_ids = embed_csv_files(claim_file_path, data_type = "claims")
    chunk_embeddings, chunks, chunk_ids = embed_csv_files(chunk_file_path, data_type = "chunks")
    FAISS_index = build_FAISS_index(chunk_embeddings)

    # returns top N relevant chunks for each query claim
    distances, indices = FAISS_index.search(claim_embeddings, top_N) # indices is 2D array, (num of entries, N indices for each query claim)
    relevant_chunks = []
    for indices_entry in indices:
        relevant_chunks.append([chunks[index] for index in indices_entry])

    # uncertainty labelling for each claim and its top N chunks using pre-trained scibert
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialise scibert model
    # change file directory for pre-trained autotokenizer and model
    scibert_tokenizer = AutoTokenizer.from_pretrained("/homes/ko25/Desktop/fyp/scibert_finetuned_model")
    scibert_model = AutoModelForSequenceClassification.from_pretrained("/homes/ko25/Desktop/fyp/scibert_finetuned_model")

    new_entries = []

    scibert_model.to(device)

    for outer_index in range(len(claims)): # outerloop to iterate through each claim
        claim = claims[outer_index]
        # inner loop for iterating through each relevant chunk for that claim
        for inner_index, relevant_chunk in enumerate(relevant_chunks[outer_index]):
            new_entry = {}
            new_entry["claim_id"] = claim_ids[outer_index]
            new_entry["claim"] = claim
            new_entry["chunk_id"] = chunk_ids[indices[outer_index][inner_index]]
            new_entry["chunk"] = relevant_chunk
            P_STANCE, label = uncertainty_label(claim, relevant_chunk, scibert_tokenizer, scibert_model, threshold = threshold, device = device)
            distance = distances[outer_index][inner_index]
            new_entry["relevance_score"] = distance
            new_entry["stance_score"] = P_STANCE
            new_entry["label"] = label
            new_entries.append(new_entry)

    # create relationshiplink.csv
    relationshiplink_df = pd.DataFrame(new_entries)
    pd.DataFrame.to_csv(relationshiplink_df, "relationshiplink.csv", index = False)

