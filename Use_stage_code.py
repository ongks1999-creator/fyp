from RAG_uncertainty_label import embed_csv_files, build_FAISS_index, uncertainty_label
import pandas as pd
import faiss
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

"""
Script for the Usage Stage:

- Runs through each query claim in query_claims.csv
- Conducts a FAISS search, matching query claim to build_claim.csv database
- Returns top 3 most relevant claims in the database
- Returns its individual relevant chunks for each retrieved claim
- Calculates chunk score and chunk weights
- Normalise chunk score with all chunk weights to get query claim score
- LLM evidence based justification
"""

def main():
    # Initialise and load files
    top_N = 15 # top N for relevant chunks
    threshold = 0.3

    claims_filepath = "/root/fyp/claims.csv"
    chunks_filepath = "/root/fyp/chunks_final.csv"
    # embeds is a list of embeddings, claims is a list of strings, ids is a list of claim_ids
    # all three rely on the same indexing
    claims_embeds, claims, claims_ids = embed_csv_files(claims_filepath, data_type = "claims")
    chunks_embeds, chunks, chunks_ids = embed_csv_files(chunks_filepath, data_type = "chunks")


    # Build FAISS index for all chunks, for semantic retrieval of claims in database to match query claim
    chunks_FAISS_index = build_FAISS_index(chunks_embeds)

    # Initialise Scibert to calculate stance score between query claim and relevant chunk
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scibert_tokenizer = AutoTokenizer.from_pretrained("/root/fyp/scibert_finetuned_model")
    scibert_model = AutoModelForSequenceClassification.from_pretrained("/root/fyp/scibert_finetuned_model")

    scibert_model.to(device)

    sources_df = pd.read_csv("/root/fyp/sources.csv")
    chunks_df = pd.read_csv("/root/fyp/chunks_final.csv")
    claims_df = pd.read_csv("/root/fyp/claims.csv")
    # build a dictionary to map ids to ids, ids to scores, faster than searching through dataframe for each iteration
    chunk_id_to_source_id = dict(zip(chunks_df["chunk_id"], chunks_df["source_id"]))
    claim_id_to_chunk_id = dict(zip(claims_df["claim_id"], claims_df["chunk_id"]))
    source_id_to_recency_score = dict(zip(sources_df["source_id"], sources_df["recency_score"]))
    source_id_to_provenance_score = dict(zip(sources_df["source_id"], sources_df["provenance_score"]))
    source_id_to_source_cred_score = dict(zip(sources_df["source_id"], sources_df["source_credibility"]))

    chunks_relevance_list, chunks_relevance_indices = chunks_FAISS_index.search(claims_embeds, top_N + 1) # search for Top N +1 relevant chunks as there may be a parent chunk to that claim 
    # chunks_relevance_indices is 2D array, (num of query claims, the indices of the 15 most relevant chunks in chunks_final.csv)   

    query_claim_list = []

    for query_claim_index, entry_indices in enumerate(chunks_relevance_indices):
        query_claim_entry = {}
        # single hop search, query claim directly to chunk database
        query_claim = claims[query_claim_index]
        query_claim_id = claims_ids[query_claim_index]
        query_claim_chunk_id = claim_id_to_chunk_id[query_claim_id]
        query_claim_source_id = chunk_id_to_source_id[query_claim_chunk_id]

        # for calculation of query claim score
        sum_chunk_scores = 0
        sum_chunk_weights = 0
        # tracks meta data and scores to give to LLM 
        source_id_list = []
        recency_score_list = []
        provenance_score_list = []
        source_credibility_list = []
        relevance_score_list = []
        P_STANCE_list = []
        filtered_chunk_indices = []
        for chunk_index in entry_indices:
           if len(filtered_chunk_indices) >= top_N: # if in the scenario no parent chunk, we set limit of selected relevant chunks to be N
                break
           if query_claim_chunk_id == chunks_ids[chunk_index]: # skip chunks that are the parent of the claim
                continue
           filtered_chunk_indices.append(chunk_index)

        # only iterate through the selected chunk indices, that have been filtered
        for chunk_index in filtered_chunk_indices:
            chunk_text = chunks[chunk_index]
            chunk_id = chunks_ids[chunk_index]
            source_id = chunk_id_to_source_id[chunk_id]
            relevant_chunk_embed = chunks_embeds[chunk_index]
            # use L2 norm to calculate relevance score between query claim and relevant chunk
            # closer to zero distance give score closer to 1, further away gives score closer to 0.5
            relevance_score = 0.5 + 0.5 * (1 / (1 + float(np.linalg.norm(claims_embeds[query_claim_index] - relevant_chunk_embed))))
            P_STANCE, _ = uncertainty_label(query_claim, chunk_text, scibert_tokenizer, scibert_model, threshold = threshold, device = device)
            source_credibility = source_id_to_source_cred_score[source_id]
            recency_score = source_id_to_recency_score[source_id]
            provenance_score = source_id_to_provenance_score[source_id]
            chunk_weight = source_credibility * recency_score * provenance_score * relevance_score
            chunk_score = P_STANCE * chunk_weight
            sum_chunk_scores += chunk_score
            sum_chunk_weights += chunk_weight
            source_id_list.append(source_id)
            recency_score_list.append(recency_score)
            provenance_score_list.append(provenance_score)
            source_credibility_list.append(source_credibility)
            relevance_score_list.append(relevance_score)
            P_STANCE_list.append(P_STANCE)

        
        query_claim_score = sum_chunk_scores / sum_chunk_weights

        query_claim_entry["claim_id"] = query_claim_id
        query_claim_entry["claim_source_id"] = query_claim_source_id
        query_claim_entry["claim"] = query_claim
        query_claim_entry["query_claim_score"] = query_claim_score

        """
        Retrieving Meta data such as the relevant chunks ids, and source ids
        """
        query_claim_entry["relevant_chunk_ids"] = [chunks_ids[chunk_index] for chunk_index in filtered_chunk_indices]
        query_claim_entry["relevant_source_ids"] = source_id_list
        query_claim_entry["recency_scores"] = recency_score_list
        query_claim_entry["provenance_scores"] = provenance_score_list
        query_claim_entry["source_credibility_scores"] = source_credibility_list
        query_claim_entry["relevance_scores"] = relevance_score_list
        query_claim_entry["stance_scores"] = P_STANCE_list
        query_claim_entry["company"] = claims_df.iloc[query_claim_index]["company"] # index df based on query claim index
        query_claim_entry["claims originating magazine"] = claims_df.iloc[query_claim_index]["magazine"]
        query_claim_list.append(query_claim_entry)
    
    query_claim_scores_df = pd.DataFrame(query_claim_list)
    pd.DataFrame.to_csv(query_claim_scores_df, "query_claim_scores.csv", index = False)


if __name__ == "__main__":
    main()