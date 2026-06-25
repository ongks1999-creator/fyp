import csv
import ast
from transformers import AutoTokenizer
import torch


def create_training_pairs(corpus_csv_file_path, claims_train_csv_file_path) -> list[tuple[str, str, str]]:
    # load both the corpus csv and claim train csv files
    with open(corpus_csv_file_path, newline = '') as csvfile:
        corpus_csv = csv.DictReader(csvfile)
        corpus_csv = list(corpus_csv) # convert dic to list of dicts

        # convert everything into a single dictionary, with key as doc id and value as abstract (evidence)
        # for quick indexing and search when iterating through claims train csv
        corpus_csv_dict = {}
        for entry in corpus_csv: # doc id converted from str to int
            # abstract entry is a string with a list inside, convert it to list form
            corpus_csv_dict[int(entry["doc_id"])] = ast.literal_eval(entry["abstract"])


    with open(claims_train_csv_file_path, newline = '') as csvfile:
        claims_train_csv = csv.DictReader(csvfile)
        claims_train_csv = list(claims_train_csv)
        training_pairs = []

        # creation of training pairs for each complete entry in claim_train csv
        for entry in claims_train_csv:

            # creation of support/contradict training pairs
            if entry["evidence_doc_id"]: # they have complete entries having evidence doc id
                index = ast.literal_eval(entry["evidence_sentences"]) # find the sentence with index of "evidence_sentences" in the abstract list
                abstract = corpus_csv_dict[int(entry["evidence_doc_id"])] # locate key-value pair in corpus csv dict
                # index the list of sentences in abstract, as evidence_sentences can be list of index
                evidence = ""
                for i in index:
                    evidence +=  abstract[i] + " "

                training_pairs.append((entry["claim"], evidence.strip(), entry["evidence_label"]))

            if entry["evidence_doc_id"] == "": # for NEI training pairs
                # only add entries if the corpus doc id is present and the abstract contains text
                cited_doc_id = ast.literal_eval(entry["cited_doc_ids"])
                if cited_doc_id:
                    abstract = corpus_csv_dict[(cited_doc_id[0])] # cited_doc_id is a list, need to index
                    if abstract:
                        evidence = abstract[0] # use first sentence as the supporting statement
                        training_pairs.append((entry["claim"], evidence, "NEI"))

    return training_pairs

def main(corpus_csv_file_path, claims_train_csv_file_path):

    labels = {"NEI": 0, "CONTRADICT": 1,"SUPPORT": 2}
    model = "allenai/scibert_scivocab_uncased"
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    training_pairs = create_training_pairs(corpus_csv_file_path, claims_train_csv_file_path)
 
    ## Tokenisation
    # use batch tokenisation by inputting three list into tokeniser, to ensure same lenth of tokens, for stacking into tensor
    claims = [pair[0] for pair in training_pairs]
    evidences = [pair[1] for pair in training_pairs]
    labels_string = [pair[2] for pair in training_pairs]
    labels = [labels[label] for label in labels_string] # convert label from str to int

    # ensure claim evidence pair is consistently under 512
    encoded_pairs = tokenizer(claims, evidences, padding = True, truncation = True, return_tensors = "pt")


if __name__ == "__main__":
    #corpus_csv_file_path = 
    #claims_train_csv_file_path = 

    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    encode = tokenizer(["sky is blue"], ["sky is blue coloured"], padding = True, truncation = True, return_tensors = "pt")
    print(encode)