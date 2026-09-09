# fyp
Masters project

Datasets built during construction stage:
1. For 22/24 Corpus (referred to as V2 in the pipeline, as V2 was ran using single hop retrieval, v1 was double hop)
- chunks_final.csv
- sources.csv
- claims.csv

2. For 24/26 Corpus
- chunks_final_2426.csv
- sources_2426.csv
- claims_2426.csv

Pre-Run Setup:
- pip install requirement.txt libraries
- set Ollama to run on port 11435
- change file directory path within each file accordingly

Code to Run in Sequence for Construction of Pipeline for 22/24 corpus:
1. Using downloaded reddit dataset, run article_filter.py to filter off non-SMR related post and extract links -> get filtered reddit articles and extracted external links per article
3. Using inherited corpus for nuclear pub sources from previous student and filtered reddit articles, run extract_paragraphs.py -> create paragraphs
4. Run link_extraction_from_url.py to extract external links from nuclear publications articles
5.  Using paragraph file, run create_chunks.py -> create chunks
6.  Combine both chunks from both nuclear pub and reddit to get chunks_final.csv
7.  using articles from nuclear pubs, reddit, and their respective external links files, run create_sources_csv.py to get sources.csv file
8.  Train and Fine-tune SciBERT:
    - create 100 entry nuclear dataset using llama_nuclear_dataset_creation.py
    - nuclear_dataset_creation_update.py to update the nuclear dataset to solve lexical     overlap issue
    - train_scibert.py to train SciBERT on sciFact
    - fine-tune scibert with scibert_finetuning.py
    - evaluate its performance with scibert_performance.py
9. Llama claim extraction and claim filtering using llama_claim_extraction.py

(Only for 24/26 corpus that required crawling)
1. Run files in webcrawl
   - webscrapper.py to collect urls from all nuclear pubs
   - scrape the urls using the respective _new.py scrapper files
   - filter articles based on time window and combine files using inherited clearning_and_merging_articles.ipynb

Use Stage Code
1. Retrieval and scoring using Use_stage_code.py

Data analysis of results with data_analysis.py
- Cochran Amitrage analysis
- Spearman correlation analysis
- Significant companies including BC correction
- Overclaimed and underclaimed companies
- Scatter plots for Supp/Cont/NEI percentage against claim count
- Forest plots

Not included in Report (large files):
- scibert trained and fine-tuned weights
- reddit post and thread files downloaded from artic shift
- 
