# fyp
Masters project

Datasets built during construction stage:
1. For 22/24 Corpus (referred to as V2 in the pipeline, as V2 was ran using single hop retrieval, v1 was double hop, hop_comparison.py compares the double hop and single hop retrieval)
- chunks_final.csv
- sources.csv
- claims.csv

2. For 24/26 Corpus
- chunks_final_2426.csv
- sources_2426.csv
- claims_2426.csv

Pre-Run Setup:
- pip install the libraries found in requirements.txt
- set Ollama to run on port 11435
- change file directory path within each file accordingly

Code to Run in Sequence for Construction of Pipeline for 22/24 corpus:
1. Using downloaded reddit dataset, run article_filter.py to filter off non-SMR related post and extract links -> get filtered reddit articles and extracted external links per article
2. Train and Fine-tune SciBERT:
    - create 100 entry nuclear dataset using llama_nuclear_dataset_creation.py
    - train_scibert.py to train SciBERT on sciFact
    - fine-tune scibert with scibert_finetuning.py
    - evaluate its performance with scibert_performance.py
3. Using inherited corpus for nuclear pub sources from previous student and filtered reddit articles, run extract_paragraphs.py -> create paragraphs
4. Run link_extraction_from_url.py
- first run to extract external links from nuclear publications articles by commenting out filter_links and edit_links_file functions, keep main_loop
- second run to filter out noisy links with main_loop commented out and filter_links and edit_links_file kept in code
5.  Using paragraph file, run create_chunks.py -> create chunks
6.  Combine both chunks from both nuclear pub and reddit to get chunks_final.csv
7.  using the combined articles from nuclear pubs and reddit, and their respective external links files, run create_sources_csv.py to get sources.csv file
8. Llama claim extraction and claim filtering using llama_claim_extraction.py, need to comment out line 51-63 and change chunks_2426 to chunks at line 74, as the smr filter was meant for 24/26 corpus
9. nuclear_dataset_creation_update.py to update the nuclear dataset to solve lexical overlap issue, re-run the scibert fine-tuning and evaluation files with filepath directed to new nuclear dataset

(Only for 24/26 corpus that required crawling)
1. Run files in webcrawl
   - webscrapper.py to collect urls from all nuclear pubs
   - scrape the urls using the respective _new.py scrapper files
   - filter articles based on time window and combine files using inherited cleaning_and_merging_articles.ipynb
2. Run llama_claim_extraction.py to limit claim extraction to chunks from smr articles only

Use Stage Code
1. Retrieval and scoring using Use_stage_code.py

Data analysis of results with data_analysis.py
- Cochran Amitrage analysis
- Spearman correlation analysis
- Significant companies including BC correction
- Overclaimed and underclaimed companies
- Scatter plots for Supp/Cont/NEI percentage against claim count
- Forest plots
* for each data analysis run, remember to change CONFIG dict and CONFIG_NAME to what is desired

Configuration Ablation Results in diff_configs folder:
- CONFIG_NAME = "baseline", smr_filter = False, CONFIG = dict(floor = 0.5)
- CONFIG_NAME = "baseline_smr_only", smr_filter = True, CONFIG = dict(floor = 0.5)
- CONFIG_NAME = "floor_0.1", smr_filter = False, CONFIG = dict(floor = 0.1)
- CONFIG_NAME = "just_stance_score", smr_filter = False, CONFIG = dict(attribute_out = "all")
- CONFIG_NAME = "relevance_out", smr_filter = False, CONFIG = dict(attribute_out = "relevance")
- CONFIG_NAME = "provenance_out", smr_filter = False, CONFIG = dict(attribute_out = "provenance")
- CONFIG_NAME = "recency_out", smr_filter = False, CONFIG = dict(attribute_out = "recency")
- CONFIG_NAME = "source_credibility_out", smr_filter = False, CONFIG = dict(attribute_out = "source credibility")
- CONFIG_NAME = "top_3", smr_filter = False, CONFIG = dict(top_N = 3)
- CONFIG_NAME = "top_5", smr_filter = False, CONFIG = dict(top_N = 5)
- CONFIG_NAME = "top_7", smr_filter = False, CONFIG = dict(top_N = 7)
- For 24/26 corpus, CONFIG_NAME = "period_2426", smr_filter = False, CONFIG = dict(floor = 0.5)


Not included in Repo (large files):
- scibert trained and fine-tuned weights
- reddit post and thread files downloaded from Artic Shift
- paragraphs_2426.csv paragraph for 24/26 files which isnt part of database
- SciFact, Fever, Dbpedia datasets which their filepaths are used in fyp_stat_analysis.py, this code was used to verify the amount of nuclear and physics data in each dataset
- 22/24 inherited article files which their filepaths are used in llama_claim_extraction.py

