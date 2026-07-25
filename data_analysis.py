import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from collections import defaultdict, Counter
import ast
from rapidfuzz import fuzz, process
from sklearn.cluster import KMeans
import numpy as np
from adjustText import adjust_text

# Understanding the distribution of query claim scores to find the thresholds to separate the final verification labels
results_df = pd.read_csv("/homes/ko25/Desktop/fyp/query_claim_scores.csv")
figure = plt.figure()
axis = figure.add_subplot(1, 1, 1)
axis.hist(pd.to_numeric(results_df["query_claim_score"]), bins = 100)
axis.set_xlabel("Query Claim Score")
axis.set_ylabel("Frequency")
plt.savefig("query_claim_score_dist.png")
plt.show()
print(pd.to_numeric(results_df["query_claim_score"]).describe())

lower_quartile = pd.to_numeric(results_df["query_claim_score"]).quantile(0.25)
upper_quartile = pd.to_numeric(results_df["query_claim_score"]).quantile(0.75)
# used interquartile range to find the thresholds
IQR = upper_quartile - lower_quartile
threshold = 1.5
support_threshold = upper_quartile + threshold * IQR
contradict_threshold = lower_quartile - threshold * IQR

def classification(score, support_threshold, contradict_threshold):
    if score >= support_threshold:
        return "SUPPORT"
    elif score <= contradict_threshold:
        return "CONTRADICT"
    else:
        return "NEI"



results_df["Verdict Label"] = results_df["query_claim_score"].apply(classification, args = (support_threshold, contradict_threshold))
# breakdown into individual companies, since each entry for results df contain a single label for one or more companies
company_dict = defaultdict(list)
for index, entry in (results_df.iterrows()):
    if pd.isna(entry["company"]) or entry["company"] is None or entry["company"].lower() == "null":
        continue # skip invalid entries
    for company in ast.literal_eval(entry["company"]): # since value given is a string, need turn to list
        if pd.isna(entry["company"]) or company is None or company.lower() == "null":
            continue
        company_dict[company].append(entry["Verdict Label"])

# company names may be duplicated due to different naming convention by LLM
# standardising company names using fuzzy matching

# we creating a mapping dict to map the similar non standard names to their standardised form
# key is the unstandardised name, value is the standardised name
# standardised company list is a list of these standardised names
mapping_dict = {}
standardised_company_list = []

for company in company_dict.keys():
    # for the first entry, when our standardised list is empty
    if not standardised_company_list:
        # we add it to standardised name and map it to itself
        standardised_company_list.append(company)
        mapping_dict[company] = company
        continue

    most_similar_name, simi_score, index = process.extractOne(company, standardised_company_list)
    # if the current company is very similar to an existing standardised name, we add it to the names that get mapped to the standardised name
    if simi_score > 80: # must be high enough to ensure moderate similarity means the same company
        mapping_dict[company] = most_similar_name
    # if it is not similar, that means it is a new company, add it to standardise name list, and map to itself
    else:
        standardised_company_list.append(company)
        mapping_dict[company] = company

finalised_company_dict = defaultdict(list)
for company, label_list in company_dict.items():
        standardised_name = mapping_dict[company]
        finalised_company_dict[standardised_name].extend(label_list)

company_list = []
for company, label_list in finalised_company_dict.items():
    entry = {}
    entry["company"] = company
    for label, count in Counter(label_list).items():
        if label not in ["SUPPORT", "CONTRADICT", "NEI"]:
            continue # to catch None, null labels
        entry[label] = count
    company_list.append(entry)

company_df = pd.DataFrame(company_list)
company_df["Total Labels"] = company_df["SUPPORT"] + company_df["CONTRADICT"] + company_df["NEI"]
top_30 = company_df.sort_values("Total Labels", ascending = False).head(30)

# Percentage of Supported Claims, Top 30 Companies
top_30["Percentage of Supported Claims"] = (top_30["SUPPORT"] / top_30["Total Labels"]) * 100
# Percentage of Contradicted Claims
top_30["Percentage of Contradicted Claims"] = (top_30["CONTRADICT"] / top_30["Total Labels"]) * 100
# Percentage of NEI Claims
top_30["Percentage of NEI Claims"] = (top_30["NEI"] / top_30["Total Labels"]) * 100
# Verifiability as we define it as Percentage Support - Percentage Contradict
top_30["Percentage of Verifiability"] = top_30["Percentage of Supported Claims"] - top_30["Percentage of Contradicted Claims"]

# K means clustering based on verifiability of claims per company
kmeans = KMeans(n_clusters = 3, random_state = 45)
top_30["cluster"] = kmeans.fit_predict(top_30[["Percentage of Verifiability", "Percentage of NEI Claims"]])
top_30[["company", "Percentage of Verifiability", "Percentage of NEI Claims", "cluster"]].sort_values("cluster").to_csv("top_30_companies_verifiability_clustering.csv", index = False)


scatter_plot = plt.figure(figsize = (20, 10))
#colours = cm.tab20(np.linspace(0, 1, len(top_30)))
scatter_veri_axis = scatter_plot.add_subplot(1, 2, 1)
scatter_veri_axis.set_xlabel("Total Labels")
scatter_veri_axis.set_ylabel("Percentage of Verifiability")
scatter_NEI_axis = scatter_plot.add_subplot(1, 2, 2)
scatter_NEI_axis.set_xlabel("Total Labels")
scatter_NEI_axis.set_ylabel("Percentage of NEI Claims")
cluster_color = {0: "red", 1: "blue", 2: "green"}
ver_axis_entries = []
NEI_axis_entries = []

for index, row in top_30.iterrows(): # label each dot
    color = cluster_color[row["cluster"]] # top 30 clustering was based on percentage verifiability and NEI features
    scatter_veri_axis.scatter(row["Total Labels"], row["Percentage of Verifiability"], c = color)
    scatter_NEI_axis.scatter(row["Total Labels"], row["Percentage of NEI Claims"], c = color)
    ver_axis_entries.append(scatter_veri_axis.text(row["Total Labels"], row["Percentage of Verifiability"], row["company"], fontsize = 15))
    NEI_axis_entries.append(scatter_NEI_axis.text(row["Total Labels"], row["Percentage of NEI Claims"], row["company"], fontsize = 15))

adjust_text(ver_axis_entries, ax = scatter_veri_axis)
adjust_text(NEI_axis_entries, ax = scatter_NEI_axis)
plt.savefig("top_30_companies_scatter.png")

# K means based on publicity and coverage per company
kmeans = KMeans(n_clusters = 3, random_state = 45)
top_30["cluster"] = kmeans.fit_predict(top_30[["Total Labels"]])
top_30[["company", "Total Labels", "cluster"]].sort_values("cluster").to_csv("top_30_companies_publicity_clustering.csv", index = False)
