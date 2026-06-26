from sklearn.metrics import classification_report, confusion_matrix
from train_scibert import create_data_pairs, validation
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter


if __name__ == "__main__":
    claim_validation_filepath = "/vol/bitbucket/ko25/scifact/claims_validation.csv"
    corpus_csv_file_path = "/vol/bitbucket/ko25/scifact/corpus_train.csv"
    claims_train_csv_file_path = "/vol/bitbucket/ko25/scifact/claims_train.csv"
    training_pairs = create_data_pairs(corpus_csv_file_path, claims_train_csv_file_path)
    training_labels = [pair[2] for pair in training_pairs]
    training_count = Counter(training_labels)
    print(training_count) # check distribution of labels in training set
    
    # load previously trained model for evaluation on validation set
    validation_pairs = create_data_pairs(corpus_csv_file_path, claim_validation_filepath)
    tokenizer = AutoTokenizer.from_pretrained("/homes/ko25/Desktop/fyp/scibert_trained_model")
    model = AutoModelForSequenceClassification.from_pretrained("/homes/ko25/Desktop/fyp/scibert_trained_model")
    labels = {"NEI": 0, "CONTRADICT": 1,"SUPPORT": 2}

    validation_claims = [pair[0] for pair in validation_pairs]
    validation_evidences = [pair[1] for pair in validation_pairs]
    validation_labels_string = [pair[2] for pair in validation_pairs]
    validation_labels = [labels[label] for label in validation_labels_string]
    validation_encoded_pairs = tokenizer(validation_claims, validation_evidences, padding = True, truncation = True, return_tensors = "pt")
    validation_dataset = TensorDataset(validation_encoded_pairs["input_ids"], validation_encoded_pairs["attention_mask"], validation_encoded_pairs["token_type_ids"], torch.tensor(validation_labels))
    validation_data_loader = DataLoader(validation_dataset, batch_size = 32, shuffle = False)

    precision, recall, f1, average_loss, model_predictions, truth_labels = validation(model, validation_data_loader, device = "cpu") # model is on cpu
    print(classification_report(truth_labels, model_predictions, target_names = ["NEI", "CONTRADICT", "SUPPORT"]))
    print(confusion_matrix(truth_labels, model_predictions))
    
