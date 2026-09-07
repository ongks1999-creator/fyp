import pandas as pd
import numpy as np
import ast

chunks = pd.read_csv("chunks_final.csv")
chunk_dic = {}
for index, entry in chunks.iterrows():
    chunk_dic[entry["chunk_id"]] = entry["chunk"]

query_claim_scores = pd.read_csv("query_claim_scores.csv")

IQR_supp_thres = 0.508338
IQR_contra_thres = -0.33487

def predict(query_claim_score, supp_thres, contra_thres):

    if query_claim_score >= supp_thres:
        return "SUPPORT"
    elif query_claim_score <= contra_thres:
        return "CONTRADICT"
    else:
        return "NEI"

query_claim_predictions = query_claim_scores["query_claim_score"].apply(predict, args = (IQR_supp_thres, IQR_contra_thres))
query_claim_scores["predicted_label"] = query_claim_predictions

# sample 10 for each predicted label
support_samples = query_claim_scores[query_claim_scores["predicted_label"] == "SUPPORT"].sample(n = 10, random_state = 50)
contradict_samples = query_claim_scores[query_claim_scores["predicted_label"] == "CONTRADICT"].sample(n = 10, random_state = 50)
NEI_samples = query_claim_scores[query_claim_scores["predicted_label"] == "NEI"].sample(n = 10, random_state = 50)

def create_entries_list(x, label):
    """
    x is dataframe of samples
    """
    entries = []
    for index, entry in x.iterrows():
        entries_dic = {}

        claim_id = entry["claim_id"]
        claim_text = entry["claim"]
        entries_dic["claim_id"] = claim_id
        entries_dic["claim_text"] = claim_text
        relevant_chunk_ids = ast.literal_eval(entry["relevant_chunk_ids"])
        stance_scores = ast.literal_eval(entry["stance_scores"])
        chunk_df = pd.DataFrame({"chunk_id": relevant_chunk_ids, "stance_score": stance_scores, "stance_score_val": np.abs(stance_scores)})

        chunk_df_sorted = chunk_df.sort_values("stance_score_val", ascending = False).head(3) # sort by top 3 chunks with highest stance, need convert to abs number

        counter = 0
        retrieved_chunks = []
        retrieved_chunks_ids = []
        stance_scores_list = []
        for index, row in chunk_df_sorted.iterrows():
            if counter == 3: # examine top 3 highest stance score chunks for manual validation
                break
            chunk_text = chunk_dic[row["chunk_id"]]
            retrieved_chunks.append(f"[CHUNK {counter + 1}] {chunk_text}")
            retrieved_chunks_ids.append(row["chunk_id"])
            stance_scores_list.append(row["stance_score"])
            counter += 1

        entries_dic["retrieved_chunks"] = retrieved_chunks
        entries_dic["retrieved_chunks_ids"] = retrieved_chunks_ids
        entries_dic["stance_scores"] = stance_scores_list
        entries_dic["predicted_label"] = label
        entries_dic["my_prediction"] = ""
        entries.append(entries_dic)

    return entries

support_list = create_entries_list(support_samples, "SUPPORT")
contradict_list = create_entries_list(contradict_samples, "CONTRADICT")
nei_list = create_entries_list(NEI_samples, "NEI")

combined_list = support_list + contradict_list + nei_list

pd.DataFrame(combined_list).to_csv("manual_validation.csv", index = False)

    



