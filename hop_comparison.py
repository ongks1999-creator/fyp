import pandas as pd
import numpy as np
import ast

# v1 are the scores using double hop retrieval, v2 is using single hop retrieval
v1 = pd.read_csv("/Users/ongkaisheng/Desktop/ImperialCollege/FYP/fyp/backup/v1/query_claim_scores_v1.csv")
v2 = pd.read_csv("/Users/ongkaisheng/Desktop/ImperialCollege/FYP/fyp/query_claim_scores.csv")

v1_avg = []
v1_claim_ids = set() # use to track claims that are shared between both variants, since v2 has twice the number of claims
# as claim csv is used entirely as query claims in v2
v2_avg = []

for _, entry in v1.iterrows():
    entry_mean = np.mean(ast.literal_eval(entry["relevance_scores"]))
    v1_avg.append(entry_mean)
    v1_claim_ids.add(entry["claim_id"])

for _, entry in v2.iterrows():
    if entry["claim_id"] in v1_claim_ids: # only include claims that are found in v1
        entry_mean = np.mean(ast.literal_eval(entry["relevance_scores"]))
        v2_avg.append(entry_mean)

v1_sd = np.std(v1_avg)
v2_sd = np.std(v2_avg)
v1_avg = np.mean(v1_avg)
v2_avg = np.mean(v2_avg)


print("v1 average:", v1_avg)
print("v2 average:", v2_avg)
print("Percentage increase:", (v2_avg - v1_avg) / v1_avg * 100)
print("v1 standard deviation:", v1_sd)
print("v2 standard deviation:", v2_sd)
print("Percentage change in standard deviation:", (v2_sd - v1_sd) / v1_sd * 100)
