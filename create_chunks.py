import csv
import pandas as pd
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
chunk_token_limit = 400

def create_chunk_from_paragraph(paragraphs_csv_file_path) -> None:
    with open(paragraphs_csv_file_path, newline = '') as csvfile:
        paragraphs_csv = csv.DictReader(csvfile)
        paragraphs_csv = list(paragraphs_csv) # convert dic to list of dicts

        # convert each token count from str to int
        for paragraph in paragraphs_csv:
            paragraph["token_count"] = int(paragraph["token_count"])

        chunk_list = [] # list of dict, used to convert to panda df
        next_start = 0 # variable to ensure paragraphs that were alr included in the chunk are skipped

        def create_chunk_entry(index_start, index_end) -> dict:
            chunk = "" # initialise empty string

            for paragraph in paragraphs_csv[index_start:index_end]:
                chunk += paragraph["text"] + " " # separate paragraph text by space
            # create a dict for each chunk entry
            chunk_entry = {}
            # creation of chunk_id based on its source_id and paragraph index
            chunk_entry["chunk_id"] = f'{index_start_source_id}_{index_start}_{index_end - 1}'
            chunk_entry["source_id"] = index_start_source_id
            chunk_entry["url"] = url
            chunk_entry["title"] = title
            chunk_entry["magazine"] = magazine
            chunk_entry["date"] = date
            chunk_entry["chunk"] = chunk
            # chunk_entry token count is the accurate token count that needs to be under 400
            # this is explained further in the chunking methodology section
            # compared to chunk_token_count which is an estimate sum of individual paragraph tokens
            chunk_entry["token_count"] = len(tokenizer.tokenize(chunk))

            return chunk_entry
        
        for index_start in range(len(paragraphs_csv)):
            if index_start < next_start: # next chunk starts from where we ended
                continue # skip to next index

            index_start_source_id = paragraphs_csv[index_start]["source_id"]
            url = paragraphs_csv[index_start]["url"]
            title = paragraphs_csv[index_start]["title"]
            magazine = paragraphs_csv[index_start]["magazine"]
            date = paragraphs_csv[index_start]["date"]
            chunk_token_count = paragraphs_csv[index_start]["token_count"]

            for index_end in range(index_start + 1, len(paragraphs_csv) + 1):
                if index_end == len(paragraphs_csv):
                    # this check ensures that if the current chunk (including the last paragraph) is under the token limit and is same source_id
                    #  we add it to the list, this check prevents index error too
                        chunk_list.append(create_chunk_entry(index_start, index_end - 1)) # need minus one as index end is alr out of range
                        next_start = index_end # assigned to ensure the next chunk starts from where we ended
                        break # break to prevent iterating to next index in inner loop

                chunk_token_count += paragraphs_csv[index_end]["token_count"]
                if chunk_token_count > chunk_token_limit or paragraphs_csv[index_end]["source_id"] != index_start_source_id:
                    # if its a different source id or exceed token limit, we add up to the previous paragraph
                    chunk_list.append(create_chunk_entry(index_start, index_end))
                    next_start = index_end
                    break

        chunk_df = pd.DataFrame(chunk_list)
        pd.DataFrame.to_csv(chunk_df, "chunks.csv", index = False)
    
    return

def verify_chunk_token_limit(chunk_csv_file_path, limit = chunk_token_limit) -> list:
    """
    Identify any chunks that have exceeded the token limit of 400.
    They will be adjusted in length during the subsequent faithfulness and stance score calculation stage with SciBERT
    Ensuring that total claim + chunk + special tokens (CLS and SEP) <= 512 tokens
    """
    with open(chunk_csv_file_path, newline = '') as csvfile:
        chunk_csv = csv.DictReader(csvfile)
        chunk_csv = list(chunk_csv)

        for chunk_entry in chunk_csv:
            chunk_entry["token_count"] = int(chunk_entry["token_count"])
            
        oversized_chunks = [chunk_entry for chunk_entry in chunk_csv if chunk_entry["token_count"] > limit]

    return oversized_chunks
            
if __name__ == "__main__":
    filepath = "/Users/ongkaisheng/Desktop/ImperialCollege/FYP/fyp/paragraphs.csv"
    create_chunk_from_paragraph(filepath)
    oversized_chunks = verify_chunk_token_limit("/Users/ongkaisheng/Desktop/ImperialCollege/FYP/fyp/chunks.csv")
    print(f"Number of oversized chunks: {len(oversized_chunks)}")
    print(f"Token size: {[chunk['token_count'] for chunk in oversized_chunks]}")

