import pandas as pd
import numpy as np
import re
from collections import defaultdict
from collections import Counter
from RAG_uncertainty_label import embed_csv_files, build_FAISS_index, uncertainty_label

nuclear_dataset = pd.read_csv("/Users/ongkaisheng/Desktop/ImperialCollege/FYP/fyp/finalised_finetune_dataset.csv")

# Diagnosis
# check for possible overlap in the claim and chunk for each entry
def check_overlap(claim: str, chunk: str):
    # for each claim, check the number of common words between claim and chunk:
    # split the claim and chunk into just words
    claim_words = set(re.split(r"\W+", claim.lower())) - {""} # to remove empty string elements after split
    chunk_words = set(re.split(r"\W+", chunk.lower())) - {""}

    if not claim_words or not chunk_words: # if claim/chunk is empty, no overlap
        return 0.0

    # use conjunction
    overlap = len(claim_words & chunk_words) / len(claim_words)

    return overlap

overlap_ranges = [0, 0.25, 0.5, 0.75, 1.0]
label_dist = defaultdict(list)

def catergorise_overlap(overlap: float):
    lower_bound = 0.0

    for i in overlap_ranges:
        if overlap > i:
            lower_bound = i
            continue
        if overlap <= i:
            upper_bound = i

            return f"{lower_bound}-{upper_bound}"

for index, entry in nuclear_dataset.iterrows():
    claim = entry["claim"]
    chunk = entry["chunk"]
    label = entry["label"]
    overlap = check_overlap(claim, chunk)
    label_dist[label].append(catergorise_overlap(overlap))

for label, bands in label_dist.items():
    band_counts = Counter(bands)
    print(f"Label: {label}")
    for band, count in band_counts.items():
        print(f"Overlap Band Range: {band}, Count: {count}")

chunks_final = pd.read_csv("/Users/ongkaisheng/Desktop/ImperialCollege/FYP/fyp/chunks_final.csv")
claims = pd.read_csv("/Users/ongkaisheng/Desktop/ImperialCollege/FYP/fyp/claims.csv")

# claims and chunks alr in our original dataset
existing_claims = set(nuclear_dataset["claim"])
existing_chunks = set(nuclear_dataset["chunk"])

# sample 300 random randoms from claims.csv
sample_size = 300
claims_sampled = claims.sample(n = sample_size, random_state = 45)


# common sets used to track additional claim and chunk entries for this additional dataset
# used for both NEI/contra/supp
selected_chunks = set()
selected_claims_chunk_ids = set() # chunk id for that claim

# taking 20 entries each for NEI/contra/supp and then manually verifying
NEI_list = []
# Manually adding entries for NEI, iterate through all chunks and claims and take those that are above 0.5 overlap

for sample_count, (claim_index, claim) in enumerate(claims_sampled.iterrows()):
    if len(NEI_list) >= 20:
            break
    # ensure we can capture overlap values from both ranges
    if sample_count > sample_size / 2: # for second half of claims we capture higher overlap values
        threshold = 0.75
    else:
        threshold = 0.5
    claim_text = claim["claim"]
    claim_chunk_id = claim["chunk_id"]
    # ensure no duplicate claims with original dataset and the additional dataset
    if claim_text in existing_claims:
        continue
    if claim_chunk_id in selected_claims_chunk_ids:
        continue
    for chunk_index, chunk in chunks_final.iterrows():
        chunk_text = chunk["chunk"]
        # ensure no duplicate chunks
        if chunk_text in existing_chunks:
            continue
        # ensure chunk is not parent of claim
        chunk_id = chunk["chunk_id"]
        if claim_chunk_id == chunk_id:
            continue
        # ensure chunk is also not inside selected set
        if chunk_id in selected_chunks:
            continue
        overlap = check_overlap(claim_text, chunk_text)
        if overlap >= threshold:
            new_entry = {"claim": claim_text, "chunk": chunk_text, "overlap": overlap, "label": "NEI"}
            NEI_list.append(new_entry)
            selected_chunks.add(chunk_id)
            selected_claims_chunk_ids.add(claim_chunk_id)
            break



SUPP_list = []
# For supporting labels, we use semantic similarity search within the chunk csv to find the top relevant chunks,
# and filter further for overlap that is below 0.75
claims_embeds, claims_text_list, claims_ids = embed_csv_files("/Users/ongkaisheng/Desktop/ImperialCollege/FYP/fyp/claims.csv", data_type = "claims")
chunk_embeds, chunks_text_list, chunks_ids = embed_csv_files("/Users/ongkaisheng/Desktop/ImperialCollege/FYP/fyp/chunks_final.csv", data_type = "chunks")
chunk_FAISS_index = build_FAISS_index(chunk_embeds)

for sample_count, (claim_index, claim) in enumerate(claims_sampled.iterrows()):
    if sample_count > sample_size / 2: # for second half of claims we capture higher overlap values
        threshold = 0.5

    else:
        threshold = 0.75

    if len(SUPP_list) >= 20:
        break
    claim_text = claim["claim"]
    claim_chunk_id = claim["chunk_id"]
    if claim_text in existing_claims:
        continue
    if claim_chunk_id in selected_claims_chunk_ids:
        continue
    claim_embed = claims_embeds[claim_index] # claim index is the index within the whole claims.csv
    _, result_indices = chunk_FAISS_index.search(np.array([claim_embed]), k = 15)
    for chunk_index in result_indices[0]: # result_indices is 2d array
        chunk_text = chunks_text_list[chunk_index]
        chunk_id = chunks_ids[chunk_index] # chunk ids is a list of chunk id strings
        if chunk_text in existing_chunks:
            continue
        if chunk_id in selected_chunks:
            continue
        if claim_chunk_id == chunk_id: # prevent the claim from belonging to parent chunk
            continue

        overlap = check_overlap(claim_text, chunk_text)

        if overlap < threshold:
            new_entry = {"claim": claim_text, "chunk": chunk_text, "overlap": overlap, "label": "SUPPORT"}
            SUPP_list.append(new_entry)
            selected_chunks.add(chunk_id)
            selected_claims_chunk_ids.add(claim_chunk_id)
            break

CONTRA_list = []
# for adding contradict labels
for sample_count, (claim_index, claim) in enumerate(claims_sampled.iterrows()):
    if sample_count > sample_size / 2:
        threshold = 0.5

    else:
        threshold = 0.25

    if len(CONTRA_list) >= 20:
        break
    claim_text = claim["claim"]
    claim_chunk_id = claim["chunk_id"]
    if claim_text in existing_claims:
        continue
    if claim_chunk_id in selected_claims_chunk_ids:
        continue
    claim_embed = claims_embeds[claim_index] # claim index is the index within the whole claims.csv
    _, result_indices = chunk_FAISS_index.search(np.array([claim_embed]), k = 15)
    for chunk_index in result_indices[0]: # result_indices is 2d array
        chunk_text = chunks_text_list[chunk_index]
        chunk_id = chunks_ids[chunk_index] # chunk ids is a list of chunk id strings
        if chunk_text in existing_chunks:
            continue
        if chunk_id in selected_chunks:
            continue
        if claim_chunk_id == chunk_id: # reject parent chunk and claim
            continue

        overlap = check_overlap(claim_text, chunk_text)

        if overlap < threshold:
            new_entry = {"claim": claim_text, "chunk": chunk_text, "overlap": overlap, "label": "CONTRADICT"}
            CONTRA_list.append(new_entry)
            selected_chunks.add(chunk_id)
            selected_claims_chunk_ids.add(claim_chunk_id)
            break
    

extra_dataset = NEI_list + SUPP_list + CONTRA_list
pd.DataFrame(extra_dataset).to_csv("/Users/ongkaisheng/Desktop/ImperialCollege/FYP/fyp/additional_nuclear_dataset.csv", index = False)