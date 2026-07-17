import csv
import ast
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import sklearn.model_selection import KFold

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
    dataset = read_csv_create_pairs(file_path)
    # Five fold CV
    kfold = KFold(n_splits = 5, shuffle = True, random_state = 45) # fix seed to ensure reproducibility

 
