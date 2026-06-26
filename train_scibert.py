import csv
import ast
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_fscore_support


def create_data_pairs(corpus_csv_file_path, claims_train_csv_file_path) -> list[tuple[str, str, str]]:
    # load both the corpus csv and claim train csv files
    with open(corpus_csv_file_path, newline = '') as csvfile:
        corpus_csv = csv.DictReader(csvfile)
        corpus_csv = list(corpus_csv) # convert dic to list of dicts

        # convert everything into a single dictionary, with key as doc id and value as abstract (evidence)
        # for quick indexing and search when iterating through claims train csv
        corpus_csv_dict = {}
        for entry in corpus_csv: # doc id converted from str to int
            # abstract entry is a string with a list inside, however, there are no comma in between entries
            # to prevent multiple sentences from being combined into one, we replace it to add comma
            # search for inverted commas with one or more whitespace, then add a comma in between
            abstract = re.sub(r"""([\'\"])(\s+)([\'\"])""", r"\1,\2\3", entry["abstract"])
            corpus_csv_dict[int(entry["doc_id"])] = ast.literal_eval(abstract)


    with open(claims_train_csv_file_path, newline = '') as csvfile:
        claims_train_csv = csv.DictReader(csvfile)
        claims_train_csv = list(claims_train_csv)
        training_pairs = []

        # creation of training pairs for each complete entry in claim_train csv
        for entry in claims_train_csv:

            # creation of support/contradict training pairs
            if entry["evidence_doc_id"]: # they have complete entries having evidence doc id
                indices = entry["evidence_sentences"].strip("[]").split() # find the sentence with index of "evidence_sentences" in the abstract list
                indices = [int(index) for index in indices] # indices was list of str, we convert to int
                abstract = corpus_csv_dict[int(entry["evidence_doc_id"])] # locate key-value pair in corpus csv dict
                # index the list of sentences in abstract, as evidence_sentences can be list of index
                evidence = ""
                for index in indices:
                    evidence +=  abstract[index] + " "

                training_pairs.append((entry["claim"], evidence.strip(), entry["evidence_label"]))

            if entry["evidence_doc_id"] == "": # for NEI training pairs
                # only add entries if the corpus doc id is present and the abstract contains text
                cited_doc_id = entry["cited_doc_ids"].strip("[]").split() # list of strings
                cited_doc_id = [int(doc_id) for doc_id in cited_doc_id] # convert to int
                if cited_doc_id:
                    abstract = corpus_csv_dict[(cited_doc_id[0])] # cited_doc_id is a list, need to index
                    if abstract:
                        evidence = abstract[0] # use first sentence as the supporting statement
                        training_pairs.append((entry["claim"], evidence, "NEI"))

    return training_pairs

def validation(model, data_loader, device):

    model.eval()
    total_count = 0
    predictions = []
    truth_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, token_type_ids, batch_labels = batch
            # shift data to same device as model
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            batch_labels = batch_labels.to(device)

            output = model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids, labels = batch_labels)
            total_loss += output.loss.item() * len(batch_labels) # loss is averaged over batch so need multiply with batch size
            prediction = output.logits.argmax(dim = -1) # the index with the largest logit value is the predicted label, along dim -1
            predictions.extend(prediction.cpu().tolist()) # convert tensor to list
            truth_labels.extend(batch_labels.cpu().tolist())
            total_count += len(batch_labels)

        precision, recall, f1, _ = precision_recall_fscore_support(truth_labels, predictions, average = "macro") # precision recall and f1 calculated for each class individually then averaged
        average_loss = total_loss / total_count
    return precision, recall, f1, average_loss

def main(corpus_csv_file_path, claims_train_csv_file_path, claims_validation_csv_file_path):

    labels = {"NEI": 0, "CONTRADICT": 1,"SUPPORT": 2}
    model_name = "allenai/scibert_scivocab_uncased" # scibert from: arXiv:1903.10676
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    training_pairs = create_data_pairs(corpus_csv_file_path, claims_train_csv_file_path)
    validation_pairs = create_data_pairs(corpus_csv_file_path, claims_validation_csv_file_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the pre-trained Scibert and add an additional classification layer
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 3)
    model = model.to(device)
 
    ## Tokenisation
    # use batch tokenisation by inputting three list into tokeniser, to ensure same lenth of tokens, for stacking into tensor
    training_claims = [pair[0] for pair in training_pairs]
    training_evidences = [pair[1] for pair in training_pairs]
    training_labels_string = [pair[2] for pair in training_pairs]
    training_labels = [labels[label] for label in training_labels_string] # convert label from str to int

    # ensure claim evidence pair is consistently under 512
    training_encoded_pairs = tokenizer(training_claims, training_evidences, padding = True, truncation = True, return_tensors = "pt")
    # tokenizer returns dic of tensors, 3 keys 
    # input_ids, attention_mask, token_type_ids are produced as defaults
    # bundling the tensors into a single dataset
    training_dataset = TensorDataset(training_encoded_pairs["input_ids"], training_encoded_pairs["attention_mask"], training_encoded_pairs["token_type_ids"], torch.tensor(training_labels)) # all tensor form
    # Turn data into batches
    training_data_loader = DataLoader(training_dataset, batch_size = 32, shuffle = True)

    # For validation set
    validation_claims = [pair[0] for pair in validation_pairs]
    validation_evidences = [pair[1] for pair in validation_pairs]
    validation_labels_string = [pair[2] for pair in validation_pairs]
    validation_labels = [labels[label] for label in validation_labels_string] # convert label from str to int
    validation_encoded_pairs = tokenizer(validation_claims, validation_evidences, padding = True, truncation = True, return_tensors = "pt")
    validation_dataset = TensorDataset(validation_encoded_pairs["input_ids"], validation_encoded_pairs["attention_mask"], validation_encoded_pairs["token_type_ids"], torch.tensor(validation_labels))
    validation_data_loader = DataLoader(validation_dataset, batch_size = 32, shuffle = False) # validation no need shuffle
    best_f1 = 0

    # Training loop
    # lr and epoch based on original scibert paper
    optimizer = torch.optim.AdamW(model.parameters(), lr = 2e-5)
    total_epochs = 4
    for epoch in range(total_epochs):
        model.train()
        for batch in training_data_loader:
            input_ids, attention_mask, token_type_ids, batch_labels = batch
            # shift data to same device as model
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            batch_labels = batch_labels.to(device)

            # Forward pass through model, passing the batch data as inputs
            output = model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids, labels = batch_labels)

            # Backprop
            loss = output.loss
            loss.backward() 
            optimizer.step()
            optimizer.zero_grad()

        precision, recall, f1, average_loss = validation(model, validation_data_loader, device)
        print(f"Ran Epoch {epoch + 1} out of {total_epochs}. F1: {f1}, Precision: {precision}, Recall: {recall}, Average Loss: {average_loss}, Loss used for backprop: {loss.item()}")
        if f1 > best_f1:
            best_f1 = f1
            print(f"New best F1 score: {best_f1}. Saving current model and tokenizer")
            model.save_pretrained("/homes/ko25/Desktop/fyp/scibert_trained_model")
            tokenizer.save_pretrained("/homes/ko25/Desktop/fyp/scibert_trained_model")


if __name__ == "__main__":
    corpus_csv_file_path = "/vol/bitbucket/ko25/scifact/corpus_train.csv"
    claims_train_csv_file_path = "/vol/bitbucket/ko25/scifact/claims_train.csv"
    claims_validation_csv_file_path = "/vol/bitbucket/ko25/scifact/claims_validation.csv"
    main(corpus_csv_file_path, claims_train_csv_file_path, claims_validation_csv_file_path)
