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
    top_N = 3 # top N for relevant claims
    threshold = 0.3

    query_claim_filepath = "/root/fyp/query_claims.csv"
    build_claim_filepath = "/root/fyp/build_claims.csv"

    # embeds is a list of embeddings, claims is a list of strings, ids is a list of claim_ids
    # all three rely on the same indexing
    query_claims_embeds, query_claims, query_claims_ids = embed_csv_files(query_claim_filepath, data_type = "claims")
    build_claims_embeds, build_claims, build_claims_ids = embed_csv_files(build_claim_filepath, data_type = "claims")

    # Build FAISS index for build claims, for semantic retrieval of claims in database to match query claim
    claims_FAISS_index = build_FAISS_index(build_claims_embeds)

    # Initialise Scibert to calculate stance score between query claim and relevant chunk
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scibert_tokenizer = AutoTokenizer.from_pretrained("/root/fyp/scibert_finetuned_model")
    scibert_model = AutoModelForSequenceClassification.from_pretrained("/root/fyp/scibert_finetuned_model")

    scibert_model.to(device)

    # Use dataframe to retrieve relevant chunks, and its respective sources
    relationship_link_df = pd.read_csv("/root/fyp/relationshiplink.csv")
    sources_df = pd.read_csv("/root/fyp/sources.csv")
    chunks_df = pd.read_csv("/root/fyp/chunks_final.csv")
    query_claims_df = pd.read_csv("/root/fyp/query_claims.csv")


    claim_relevance_list, claim_relevance_indices = claims_FAISS_index.search(query_claims_embeds, top_N) # search for Top 3 relevant claims 
    # claim_relevance_indices is 2D array, (num of query claims, the indices of the 3 most relevant claims in build_claims.csv)   

    # for encoding relevant chunks, to calculate relevance score with query claim
    encoding_model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1", device = device)

    query_claim_list = []

    for query_claim_index, entry_indices in enumerate(claim_relevance_indices):
        query_claim_entry = {}
        # index the build_claims_ids to get the claim_ids of the relevant claims
        relevant_claim_ids = [build_claims_ids[index] for index in entry_indices]
        query_claim = query_claims[query_claim_index]
        # rows of relationshiplink data that contain the claim_ids 
        relevant_chunks_entries = relationship_link_df[relationship_link_df["claim_id"].isin(relevant_claim_ids)]

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
    
        for _, relevant_chunk in relevant_chunks_entries.iterrows(): # dataframe iteration
            # since relationshiplink csv does not contain source_id, we manually obtain it, returns a series
            source_id = chunks_df[chunks_df["chunk_id"] == relevant_chunk["chunk_id"]]["source_id"].values[0]
            relevant_chunk_embed = encoding_model.encode(relevant_chunk["chunk"])
            # use L2 norm to calculate relevance score between query claim and relevant chunk
            # closer to zero distance give score closer to 1, further away gives score closer to 0.5
            relevance_score = 0.5 + 0.5 * (1 / (1 + float(np.linalg.norm(query_claims_embeds[query_claim_index] - relevant_chunk_embed))))
            P_STANCE, _ = uncertainty_label(query_claim, relevant_chunk["chunk"], scibert_tokenizer, scibert_model, threshold = threshold, device = device)
            source_credibility = float(sources_df[sources_df["source_id"] == source_id]["source_credibility"].values[0])
            recency_score = float(sources_df[sources_df["source_id"] == source_id]["recency_score"].values[0])
            provenance_score = float(sources_df[sources_df["source_id"] == source_id]["provenance_score"].values[0])
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

        query_claim_entry["claim_id"] = query_claims_ids[query_claim_index]
        query_claim_entry["claim"] = query_claim
        query_claim_entry["query_claim_score"] = query_claim_score

        """
        Retrieving Meta data such as the relevant claim ids, its respective chunks ids, and source ids
        """
        query_claim_entry["relevant_claim_ids"] = relevant_claim_ids
        query_claim_entry["relevant_chunk_ids"] = relevant_chunks_entries["chunk_id"].tolist()
        query_claim_entry["relevant_source_ids"] = source_id_list
        query_claim_entry["recency_scores"] = recency_score_list
        query_claim_entry["provenance_scores"] = provenance_score_list
        query_claim_entry["source_credibility_scores"] = source_credibility_list
        query_claim_entry["relevance_scores"] = relevance_score_list
        query_claim_entry["stance_scores"] = P_STANCE_list
        query_claim_entry["company"] = query_claims_df.iloc[query_claim_index]["company"] # index df based on query claim index
        query_claim_list.append(query_claim_entry)
    
    query_claim_scores_df = pd.DataFrame(query_claim_list)
    pd.DataFrame.to_csv(query_claim_scores_df, "query_claim_scores.csv", index = False)


if __name__ == "__main__":
    main()