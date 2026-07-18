import csv
import ast
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.model_selection import KFold

def read_csv_create_pairs(file_path: str) -> list:

    data_pairs = []
    with open(file_path, newline = '') as csvfile:
        dataset = csv.DictReader(csvfile)
        for row in dataset:
            data_pairs.append((row["claim"], row["chunk"], row["label"]))

    return data_pairs

def validation(model, data_loader, device, threshold):
    """
    validation code from training scibert code, but added a threshold parameter, and a decision criteria for the overall label per prediction
    """

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
            probabilities = torch.softmax(output.logits, dim = -1)  # use softmax activation to get probabilities for each label
            for probability in probabilities:
                P_NEI, P_CONTRADICT, P_SUPPORT = probability[0].item(), probability[1].item(), probability[2].item()
                P_STANCE = P_SUPPORT - P_CONTRADICT # stance score definition
                if P_STANCE > threshold: # if stance score is above the threshold, the label is SUPPORT
                    predictions.append(2)

                elif P_STANCE < -threshold: # if less than neg threshold is CONTRADICT
                    predictions.append(1)

                else:
                    predictions.append(0)  # else NEI labels

            truth_labels.extend(batch_labels.cpu().tolist())
            total_count += len(batch_labels)

        precision, recall, f1, _ = precision_recall_fscore_support(truth_labels, predictions, average = "macro") # precision recall and f1 calculated for each class individually then averaged
        average_loss = total_loss / total_count
    return precision, recall, f1, average_loss, predictions, truth_labels


def main(file_path: str):

    labels = {"NEI": 0, "CONTRADICT": 1,"SUPPORT": 2}
    model_name = "allenai/scibert_scivocab_uncased" # scibert from: arXiv:1903.10676
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = read_csv_create_pairs(file_path) # dataset which is a list of tuples
    # Five fold CV
    kfold = KFold(n_splits = 5, shuffle = True, random_state = 45) # fix seed to ensure reproducibility

    # threshold is a magnitude value
    thresholds = [value for value in np.arange(0.1,1,0.1)]

    # for tracking threshold that gives the best f1 score
    threshold_best_f1 = 0
    best_threshold = 0

    for threshold in thresholds:
        f1_scores = [] # for assessing the average f1 score across the five folds for that threshold
        # conduct 5 fold CV
        for fold_index, (train_index, validate_index) in enumerate(kfold.split(dataset)):
            print(f"Starting fold: {fold_index + 1} for threshold: {threshold}")
            # train and val index are lists of indices 
            train_data = [dataset[index] for index in train_index]
            validate_data = [dataset[index] for index in validate_index]

            # separate the data into three separate list for claims, chunk and label
            train_claim_data = [data[0] for data in train_data]
            train_chunk_data = [data[1] for data in train_data]
            train_label_data = [data[2] for data in train_data]
            # convert string labels into int for Tensor dataset
            train_labels = [labels[label] for label in train_label_data] # convert label from str to int

            # do for validation data
            validate_claim_data = [data[0] for data in validate_data]
            validate_chunk_data = [data[1] for data in validate_data]
            validate_label_data = [data[2] for data in validate_data]
            validate_labels = [labels[label] for label in validate_label_data]
            # tokenise, same training steps as train_scibert script
            train_encoded_pairs = tokenizer(train_claim_data, train_chunk_data, padding = True, truncation = True, return_tensors = "pt")
            training_dataset = TensorDataset(train_encoded_pairs["input_ids"], train_encoded_pairs["attention_mask"], train_encoded_pairs["token_type_ids"], torch.tensor(train_labels))
            # batching the training data, use batchsize 10 to reduce memory use
            train_data_loader = DataLoader(training_dataset, batch_size = 10, shuffle = True)

            # tokenise and batch for validation data
            validate_encoded_pairs = tokenizer(validate_claim_data, validate_chunk_data, padding = True, truncation = True, return_tensors = "pt")
            validate_dataset = TensorDataset(validate_encoded_pairs["input_ids"], validate_encoded_pairs["attention_mask"], validate_encoded_pairs["token_type_ids"], torch.tensor(validate_labels))
            validate_data_loader = DataLoader(validate_dataset, batch_size = 10, shuffle = False) # validation no need shuffles

            # load pre-trained scibert model with scifact
            model = AutoModelForSequenceClassification.from_pretrained("/homes/ko25/Desktop/fyp/scibert_trained_model", num_labels = 3).to(device)

            # Training loop, same code as train_scibert.py
            # lr and epoch based on original scibert paper
            optimizer = torch.optim.AdamW(model.parameters(), lr = 2e-5)
            loss_function = torch.nn.CrossEntropyLoss() # weights not used as classes are roughly balanced

            total_epochs = 20
            early_stop_counter = 0
            early_stop_limit = 2 # early stopping
            best_validation_loss = float('inf')
            best_f1 = 0 # tracking the best f1 for that fold

            for epoch in range(total_epochs):
                model.train()
                for batch in train_data_loader:
                    input_ids, attention_mask, token_type_ids, batch_labels = batch
                    # shift data to same device as model
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    token_type_ids = token_type_ids.to(device)
                    batch_labels = batch_labels.to(device)

                    # Forward pass through model, passing the batch data as inputs
                    output = model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids, labels = batch_labels)

                    # Loss and Backprop
                    loss = loss_function(output.logits, batch_labels)
                    loss.backward() 
                    optimizer.step()
                    optimizer.zero_grad()
                # calculate validation results for the particular threshold
                precision, recall, f1, average_loss, _, _ = validation(model, validate_data_loader, device, threshold)
                print(f"Ran Epoch {epoch + 1} out of {total_epochs}. F1: {f1}, Precision: {precision}, Recall: {recall}, Average Loss: {average_loss}, Loss used for backprop: {loss.item()}")

                if f1 > best_f1: # saving the best f1 score for that epoch, to be used as the f1 score for that fold
                    best_f1 = f1
                    print(f"New best F1 score: {best_f1} for fold {fold_index + 1}")

                # Early stopping
                if average_loss < best_validation_loss:
                    best_validation_loss = average_loss
                    early_stop_counter = 0 # reset early stop limit

                else:
                    early_stop_counter += 1
                    if early_stop_counter >= early_stop_limit:
                        print(f"Early stopping due to no improvement in validation loss. Current validation loss: {average_loss}")
                        break

            f1_scores.append(best_f1) # append the best f1 score achieved for that specific number of epochs for that particular fold
            
            # faced memory issue during training, had to remove model and clear cache
            del model
            torch.cuda.empty_cache()
        # average the f1 scores for all five folds, to get the f1 score for that threshold
        average_f1 = sum(f1_scores) / len(f1_scores)
        print(f"For Threshold: {threshold}, Average F1 score of: {average_f1}")
        if average_f1 > threshold_best_f1:
            threshold_best_f1 = average_f1
            best_threshold = threshold
    print(f"Overall highest F1 score of: {threshold_best_f1} achieved at threshold: {best_threshold}")

def train_final_model(file_path: str, best_threshold: float):
    """
    After conduting the five fold CV, we found the best threshold value that produced the highest F1 score
    We will now train the scibert on all 100 entries"""
    labels = {"NEI": 0, "CONTRADICT": 1,"SUPPORT": 2}
    model_name = "allenai/scibert_scivocab_uncased" # scibert from: arXiv:1903.10676
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = read_csv_create_pairs(file_path) # dataset which is a list of tuples

    # separate the data into three separate list for claims, chunk and label
    train_claim_data = [data[0] for data in dataset]
    train_chunk_data = [data[1] for data in dataset]
    train_label_data = [data[2] for data in dataset]
    # convert string labels into int for Tensor dataset
    train_labels = [labels[label] for label in train_label_data] # convert label from str to int

    # tokenise, same training steps as train_scibert script
    train_encoded_pairs = tokenizer(train_claim_data, train_chunk_data, padding = True, truncation = True, return_tensors = "pt")
    training_dataset = TensorDataset(train_encoded_pairs["input_ids"], train_encoded_pairs["attention_mask"], train_encoded_pairs["token_type_ids"], torch.tensor(train_labels))
    # batching the training data, use batchsize 10 to reduce memory use
    train_data_loader = DataLoader(training_dataset, batch_size = 10, shuffle = True)

    # load pre-trained scibert model with scifact
    model = AutoModelForSequenceClassification.from_pretrained("/homes/ko25/Desktop/fyp/scibert_trained_model", num_labels = 3).to(device)

    # Training loop, same code as train_scibert.py
    # lr and epoch based on original scibert paper
    optimizer = torch.optim.AdamW(model.parameters(), lr = 2e-5)
    loss_function = torch.nn.CrossEntropyLoss() # weights not used as classes are roughly balanced

    total_epochs = 3 # based on original scibert paper

    for epoch in range(total_epochs):
        model.train()
        for batch in train_data_loader:
            input_ids, attention_mask, token_type_ids, batch_labels = batch
            # shift data to same device as model
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            batch_labels = batch_labels.to(device)

            # Forward pass through model, passing the batch data as inputs
            output = model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids, labels = batch_labels)

            # Loss and Backprop
            loss = loss_function(output.logits, batch_labels)
            loss.backward() 
            optimizer.step()
            optimizer.zero_grad()

    model.save_pretrained("/homes/ko25/Desktop/fyp/scibert_finetuned_model")
    tokenizer.save_pretrained("/homes/ko25/Desktop/fyp/scibert_finetuned_model")

if __name__ == "__main__":
    file_path = "/homes/ko25/Desktop/fyp/finalised_finetune_dataset.csv"
    #main(file_path)
    train_final_model(file_path, best_threshold = 0.3)