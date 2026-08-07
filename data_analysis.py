import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from collections import defaultdict, Counter
import ast
from rapidfuzz import fuzz, process, utils
from sklearn.cluster import KMeans
import numpy as np
from adjustText import adjust_text
import unicodedata
import forestplot as fp
from scipy import stats
from statsmodels.stats.contingency_tables import Table
from statsmodels.stats.multitest import multipletests

# Understanding the distribution of query claim scores to find the thresholds to separate the final verification labels
results_df = pd.read_csv("/Users/ongkaisheng/Desktop/ImperialCollege/FYP/fyp/query_claim_scores.csv")
figure = plt.figure()
axis = figure.add_subplot(1, 1, 1)
axis.hist(pd.to_numeric(results_df["query_claim_score"]), bins = 100)
axis.set_xlabel("Query Claim Score")
axis.set_ylabel("Frequency")
plt.savefig("query_claim_score_dist_v2.png")
plt.show()
print(pd.to_numeric(results_df["query_claim_score"]).describe())

lower_quartile = pd.to_numeric(results_df["query_claim_score"]).quantile(0.25)
upper_quartile = pd.to_numeric(results_df["query_claim_score"]).quantile(0.75)

def threshold_calculation(dataframe, threshold = 1.5):
    lower_quartile = pd.to_numeric(dataframe).quantile(0.25)
    upper_quartile = pd.to_numeric(dataframe).quantile(0.75)

    # used interquartile range to find the thresholds
    IQR = upper_quartile - lower_quartile
    threshold = 1.5
    support_threshold = upper_quartile + threshold * IQR
    contradict_threshold = lower_quartile - threshold * IQR

    return support_threshold, contradict_threshold

support_threshold, contradict_threshold = threshold_calculation(results_df["query_claim_score"])

def classification(score, support_threshold, contradict_threshold):
    if score >= support_threshold:
        return "SUPPORT"
    elif score <= contradict_threshold:
        return "CONTRADICT"
    else:
        return "NEI"



results_df["Verdict Label"] = results_df["query_claim_score"].apply(classification, args = (support_threshold, contradict_threshold))
# grouping labels within nuclear publications group vs reddit group
results_df["Publication Group"] = np.where(results_df["claims originating magazine"] == "Reddit", "Group_Reddit", "Group_Nuclear_Publications")

label_counts = results_df["Verdict Label"].value_counts()
# Label Statistical Breakdown
for label in ["SUPPORT", "CONTRADICT", "NEI"]:
    percentage = (label_counts[label] / len(results_df)) * 100
    print(f"{label} Percentage: {percentage:.3f}%")

company_dict_by_group = defaultdict(list) # key is tuple of (company name, group type)
# breakdown into individual companies, since each entry for results df contain a single label for one or more companies
company_dict = defaultdict(list)
for index, entry in (results_df.iterrows()):
    if pd.isna(entry["company"]) or entry["company"] is None or entry["company"].lower() == "null":
        continue # skip invalid entries
    for company in ast.literal_eval(entry["company"]): # since value given is a string, need turn to list
        if pd.isna(entry["company"]) or company is None or company.lower() == "null":
            continue
        # normalise to unicode characters to ensure better matching
        company_normalised = unicodedata.normalize("NFC", company)
        company_dict[company_normalised].append(entry["Verdict Label"])
        # add separately for each company under each publication groups
        company_dict_by_group[(company_normalised, entry["Publication Group"])].append(entry["Verdict Label"])

# company names may be duplicated due to different naming convention by LLM
# standardising company names using fuzzy matching

# we creating a mapping dict to map the similar non standard names to their standardised form
# key is the unstandardised name, value is the standardised name
# standardised company list is a list of these standardised names
mapping_dict = {}
standardised_company_list = []
antimerge_names = {unicodedata.normalize("NFC", "Rosenergoatom"),
                   unicodedata.normalize("NFC", "Argonne National Laboratory"),
                   unicodedata.normalize("NFC", "UK Atomic Energy Authority (UKAEA)"),
                   unicodedata.normalize("NFC", "UK Atomic Energy Authority"),
                   unicodedata.normalize("NFC", "AEM-Technologies"),
                   unicodedata.normalize("NFC", "Korea Atomic Energy Research Institute"),
                   unicodedata.normalize("NFC", "TVO Nuclear Services"),
                   unicodedata.normalize("NFC", "Xcel Energy"),
                   unicodedata.normalize("NFC", "BWX Technologies"),
                   unicodedata.normalize("NFC", "Kyushu Electric Power Co"),
                   unicodedata.normalize("NFC", "Centrus Energy"),
                   unicodedata.normalize("NFC", "TC Energy"),
                   unicodedata.normalize("NFC", "China Power Investment Corporation"),
                   unicodedata.normalize("NFC", "GS Energy"),
                   unicodedata.normalize("NFC", "IHI Corporation"),
                   unicodedata.normalize("NFC", "NAC International"),
                   unicodedata.normalize("NFC", "Department of Energy"),
                   unicodedata.normalize("NFC", "X-Energy Canada"),
                   unicodedata.normalize("NFC", "X-energy, LLC"),
                   unicodedata.normalize("NFC", "X-energy Canada"),
                   unicodedata.normalize("NFC", "Nuclearelectrica"),
                   unicodedata.normalize("NFC", "Hitachi"),
                   unicodedata.normalize("NFC", "Laurentis Energy Partners"),
                   unicodedata.normalize("NFC", "SHINE Technologies"),
                   unicodedata.normalize("NFC", "China Nuclear Industry 22 Construction Company"),
                   unicodedata.normalize("NFC", "China Nuclear Industry 24 Construction Company"),
                   unicodedata.normalize("NFC", "Natura Resources"),
                   unicodedata.normalize("NFC", "Onet Technologies"),
                   unicodedata.normalize("NFC", "Hungarian Atomic Energy Authority"),
                   unicodedata.normalize("NFC", "Entergy Corporation"),
                   unicodedata.normalize("NFC", "Cameco Corporation"),
                   unicodedata.normalize("NFC", "Kepco Engineering & Construction"),
                   unicodedata.normalize("NFC", "China Nuclear Engineering Corporation"),
                   unicodedata.normalize("NFC", "State Power Investment Corporation"),
                   }

for company in company_dict.keys():
    # to prevent different companies from merging into one
    if company in antimerge_names:
        standardised_company_list.append(company)
        mapping_dict[company] = company
        continue
    # for the first entry, when our standardised list is empty
    if not standardised_company_list:
        # we add it to standardised name and map it to itself
        standardised_company_list.append(company)
        mapping_dict[company] = company
        continue
    # used utils default to lower case to prevent mismatch due to case sensitivity, realised company names are the same but did not match due to case sensitivity
    # changed scorer to token_sort_ratio to capture same name but different sequence
    most_similar_name, simi_score, index = process.extractOne(company, standardised_company_list, scorer = fuzz.token_sort_ratio, processor = utils.default_process)
    # if the current company is very similar to an existing standardised name, we add it to the names that get mapped to the standardised name
    # ensuring all acronyms have 100% match regardless of them matching but with different sequence
    if (simi_score > 80 and len(utils.default_process(company)) > 5) or simi_score == 100: # must be high enough to ensure moderate similarity means the same company
        mapping_dict[company] = most_similar_name
    # if it is not similar, that means it is a new company, add it to standardise name list, and map to itself
    else:
        standardised_company_list.append(company)
        mapping_dict[company] = company

# checking which companies merged, sorting companies based on the number of claims they have before the merge
# as merging this companies can cause the overall ranking to change more significantly
merging_list = []
for company, standardised_name in mapping_dict.items():
    if company != standardised_name:
        merging_list.append({"Company": company, "Merged with": standardised_name, "Number of Claims Belonging to that Company Before Merge": len(company_dict[company])})
merging_df = pd.DataFrame(merging_list)
merging_df = merging_df.sort_values("Number of Claims Belonging to that Company Before Merge", ascending = False)
merging_df.to_csv("merging_companies_v2.csv", index = False)

# alias names identified after obtaining initial results
alias_names = {unicodedata.normalize("NFC", "Fluor Enterprises"): "Fluor",
               unicodedata.normalize("NFC", "Fluor Federal Services"): "Fluor", 
               unicodedata.normalize("NFC", "GE Hitachi Nuclear Energy"): "GE-Hitachi",
               unicodedata.normalize("NFC", "Westinghouse Electric Company"): "Westinghouse",
               unicodedata.normalize("NFC", "Westinghouse Electric Sweden"): "Westinghouse",
               unicodedata.normalize("NFC", "Westinghouse Electric Company UK"): "Westinghouse",
               unicodedata.normalize("NFC", "CNNC"): "China National Nuclear Corporation",
               unicodedata.normalize("NFC", "OPG"): "Ontario Power Generation",
               unicodedata.normalize("NFC", "NuScale Power"): "NuScale",
               unicodedata.normalize("NFC", "Holtec International"): "Holtec",
               unicodedata.normalize("NFC", "MoltexFLEX"): "Moltex",
               unicodedata.normalize("NFC", "Moltex Energy"): "Moltex",
               unicodedata.normalize("NFC", "Moltex Energy Canada"): "Moltex",
               unicodedata.normalize("NFC", "Moltex Energy Limited"): "Moltex",
               unicodedata.normalize("NFC","ARC Canada"): "ARC",
               unicodedata.normalize("NFC","ARC Clean Energy Canada"): "ARC",
               unicodedata.normalize("NFC","ARC Clean Technology"): "ARC",
               unicodedata.normalize("NFC", "KHNP"): "Korea hydro and Nuclear Power (KHNP)",
               unicodedata.normalize("NFC", "GEH"): "GE-Hitachi",
               unicodedata.normalize("NFC", "INL"): "Idaho National Laboratory",
               unicodedata.normalize("NFC", "EDF Energy"): "EDF",
               unicodedata.normalize("NFC", "Flibe Energy"): "Flibe",
               unicodedata.normalize("NFC", "NANO Nuclear"): "NANO",
               unicodedata.normalize("NFC", "Kairos Power"): "Kairos",
               unicodedata.normalize("NFC", "X-energy"): "X-Energy",
               unicodedata.normalize("NFC", "X-Energy Reactor Company"): "X-Energy",
               unicodedata.normalize("NFC", "X-energy Reactor Company"): "X-Energy",
               unicodedata.normalize("NFC", "BWXT Advanced Technologies LLC"): "BWXT",
               unicodedata.normalize("NFC", "BWX Technologies"): "BWXT",
               unicodedata.normalize("NFC", "BWXT Canada"): "BWXT",
               unicodedata.normalize("NFC", "BWXT Medical"): "BWXT",
               unicodedata.normalize("NFC", "BWXT Technical Services Group"): "BWXT",
               unicodedata.normalize("NFC", "UltraSafe Nuclear"): "USNC",
               unicodedata.normalize("NFC", "Ultra Safe Nuclear Corp (USNC)"): "USNC",
               unicodedata.normalize("NFC", "Ultra Safe Nuclear"): "USNC",
               unicodedata.normalize("NFC", "Ultra Safe Nuclear Corporation"): "USNC",
               unicodedata.normalize("NFC", "ThorCon International"): "ThorCon",
               unicodedata.normalize("NFC", "Thorcon"): "ThorCon",
               unicodedata.normalize("NFC", "General Atomics Electromagnetic Systems (GA-EMS)"): "General Atomics",
               unicodedata.normalize("NFC", "Yellowcake PLC"): "Yellow Cake",
               unicodedata.normalize("NFC", "ČEZ"): "CEZ",
               unicodedata.normalize("NFC", "Oak Ridge National Laboratory (ORNL)"): "ORNL",
               unicodedata.normalize("NFC", "Oak Ridge National Laboratory"): "ORNL",
               unicodedata.normalize("NFC", "CNL"): "Canadian Nuclear Laboratories (CNL)"}
# normalise it for unicode, same as mapping_dict keys

# add mapping for names under alias name
for company, standardised_name in mapping_dict.items():
    if standardised_name in alias_names.keys():
        mapping_dict[company] = alias_names[standardised_name]

# extending label list for unstandardise company to that one standardised company, for the two groups
finalised_company_dict_by_group = defaultdict(list) # dict with keys of (standardised comp name, group type), value is list of claims for that comp
for (company, group), label_list in company_dict_by_group.items():
        standardised_name = mapping_dict[company]
        finalised_company_dict_by_group[standardised_name, group].extend(label_list)

# count labels for each company, that has standardised name as the key, based on group
company_list_by_group = [] # list with element as dict with key of comp name, and val is dict of claim count
for (company, group), label_list in finalised_company_dict_by_group.items():
    entry = {}
    entry["company"] = company
    entry["Publication Group"] = group
    for label, count in Counter(label_list).items():
        if label not in ["SUPPORT", "CONTRADICT", "NEI"]:
            continue # to catch None, null labels
        entry[label] = count
    company_list_by_group.append(entry)

## Analysing Top Companies (above respective min claims) for Nuclear Publication Group and Reddit Group
min_claims = 10 # only used to filer top comp by group, as reddit group likely has lower general claims per comp
company_df_by_group = pd.DataFrame(company_list_by_group)
# Total Claims for that company under that group, one company can have two entries as it is covered by the two publication
company_df_by_group["Total Labels"] = company_df_by_group[["SUPPORT", "CONTRADICT", "NEI"]].fillna(0).sum(axis = 1)
top_comp_by_group = company_df_by_group[company_df_by_group["Total Labels"] >= min_claims].sort_values("Total Labels", ascending = False)
company_df_by_group.sort_values("company").to_csv("company_names_by_group_v2.csv", index = False)

# have to fillna for all labels as some have NaN
top_comp_by_group["SUPPORT"] = top_comp_by_group["SUPPORT"].fillna(0)
top_comp_by_group["CONTRADICT"] = top_comp_by_group["CONTRADICT"].fillna(0)
top_comp_by_group["NEI"] = top_comp_by_group["NEI"].fillna(0)
top_comp_by_group["Percentage of Supported Claims"] = (top_comp_by_group["SUPPORT"] / top_comp_by_group["Total Labels"]) * 100
top_comp_by_group["Percentage of Contradicted Claims"] = (top_comp_by_group["CONTRADICT"] / top_comp_by_group["Total Labels"]) * 100
top_comp_by_group["Percentage of NEI Claims"] = (top_comp_by_group["NEI"] / top_comp_by_group["Total Labels"]) * 100
top_comp_by_group["Percentage of Verifiability"] = top_comp_by_group["Percentage of Supported Claims"] - top_comp_by_group["Percentage of Contradicted Claims"]
top_comp_by_group[["company", "Publication Group", "SUPPORT", "CONTRADICT", "NEI", "Total Labels", "Percentage of Supported Claims", "Percentage of Contradicted Claims", "Percentage of NEI Claims", "Percentage of Verifiability"]].sort_values("Publication Group").to_csv("top_companies_verifiability_clustering_by_group_v2.csv", index = False)


# extending label list for all unstandardised company names to that one standardised company name, for the overall database
finalised_company_dict = defaultdict(list)
for company, label_list in company_dict.items():
        standardised_name = mapping_dict[company]
        finalised_company_dict[standardised_name].extend(label_list)

# counting the labels for each of the final company, for overall database
company_list = []
for company, label_list in finalised_company_dict.items():
    entry = {}
    entry["company"] = company
    for label, count in Counter(label_list).items():
        if label not in ["SUPPORT", "CONTRADICT", "NEI"]:
            continue # to catch None, null labels
        entry[label] = count
    company_list.append(entry)

min_claims = 30 # to remove companies with too little claims
company_df = pd.DataFrame(company_list)
company_df.sort_values("company").to_csv("company_names_v2.csv", index = False)
# to catch companies that have NaN on either of the labels, as sum of NaN and int give NaN
company_df["Total Labels"] = company_df[["SUPPORT", "CONTRADICT", "NEI"]].fillna(0).sum(axis = 1)
top_comp = company_df[company_df["Total Labels"] >= min_claims].sort_values("Total Labels", ascending = False)

# have to fillna for all labels as some have NaN, top comp has the original label values from company df
# Percentage of Supported Claims
top_comp["Percentage of Supported Claims"] = (top_comp["SUPPORT"].fillna(0) / top_comp["Total Labels"]) * 100
# Percentage of Contradicted Claims
top_comp["Percentage of Contradicted Claims"] = (top_comp["CONTRADICT"].fillna(0) / top_comp["Total Labels"]) * 100
# Percentage of NEI Claims
top_comp["Percentage of NEI Claims"] = (top_comp["NEI"].fillna(0) / top_comp["Total Labels"]) * 100
# Verifiability as we define it as Percentage Support - Percentage Contradict
top_comp["Percentage of Verifiability"] = top_comp["Percentage of Supported Claims"] - top_comp["Percentage of Contradicted Claims"]

# K means clustering based on verifiability of claims per company
kmeans = KMeans(n_clusters = 3, random_state = 45)
top_comp["cluster"] = kmeans.fit_predict(top_comp[["Percentage of Verifiability", "Percentage of NEI Claims"]])
top_comp[["company", "Percentage of Verifiability", "Percentage of NEI Claims", "cluster"]].sort_values("cluster").to_csv("top_companies_verifiability_clustering_v2.csv", index = False)


scatter_plot = plt.figure(figsize = (20, 10))
scatter_veri_axis = scatter_plot.add_subplot(1, 2, 1)
scatter_veri_axis.set_xlabel("Total Labels")
scatter_veri_axis.set_ylabel("Percentage of Verifiability")
scatter_NEI_axis = scatter_plot.add_subplot(1, 2, 2)
scatter_NEI_axis.set_xlabel("Total Labels")
scatter_NEI_axis.set_ylabel("Percentage of NEI Claims")
cluster_color = {0: "red", 1: "blue", 2: "green"}
ver_axis_entries = []
NEI_axis_entries = []

for index, row in top_comp.iterrows(): # label each dot
    color = cluster_color[row["cluster"]] # top companies clustering was based on percentage verifiability and NEI features
    scatter_veri_axis.scatter(row["Total Labels"], row["Percentage of Verifiability"], c = color)
    scatter_NEI_axis.scatter(row["Total Labels"], row["Percentage of NEI Claims"], c = color)
    ver_axis_entries.append(scatter_veri_axis.text(row["Total Labels"], row["Percentage of Verifiability"], row["company"], fontsize = 15))
    NEI_axis_entries.append(scatter_NEI_axis.text(row["Total Labels"], row["Percentage of NEI Claims"], row["company"], fontsize = 15))

adjust_text(ver_axis_entries, ax = scatter_veri_axis)
adjust_text(NEI_axis_entries, ax = scatter_NEI_axis)
plt.savefig("top_companies_scatter_v2.png")

# K means based on publicity and coverage per company
kmeans = KMeans(n_clusters = 3, random_state = 45)
top_comp["cluster"] = kmeans.fit_predict(top_comp[["Total Labels"]])
top_comp[["company", "Total Labels", "cluster"]].sort_values("cluster").to_csv("top_companies_publicity_clustering_v2.csv", index = False)


# Comparison to Tiril's
# key is the company name in my data base, value is her company name in her notation and the maturity
# not inside my dataset babcox and wilcox and berkeley
comparison_company = {"Framatome": "Mature",
                      "Westinghouse": "Mature",
                     "NuScale": "Moderate Maturity",
                     "Holtec": "Moderate Maturity",
                     "GE-Hitachi": "Moderate Maturity",
                     "ARC": "Moderate Maturity",
                     "Moltex": "Moderate Maturity",
                     "ThorCon": "Least Mature",
                     "USNC": "Least Mature",
                     "General Atomics": "Least Mature",
                     "Flibe": "Least Mature",
                     "NANO": "Least Mature",
                     "Oklo": "Least Mature",
                     "Kairos": "Least Mature",
                     "BWXT": "Least Mature",
                     "X-Energy": "Least Mature",
                     "ORNL": "Least Mature",
                     "TerraPower": "Least Mature"
                     }

comparison_company_df = company_df[company_df["company"].isin(comparison_company.keys())]
comparison_company_df["Percentage of Supported Claims"] = (comparison_company_df["SUPPORT"].fillna(0) / comparison_company_df["Total Labels"]) * 100
comparison_company_df["Percentage of Contradicted Claims"] = (comparison_company_df["CONTRADICT"].fillna(0) / comparison_company_df["Total Labels"]) * 100
comparison_company_df["Percentage of NEI Claims"] = (comparison_company_df["NEI"].fillna(0) / comparison_company_df["Total Labels"]) * 100
comparison_company_df["Percentage of Verifiability"] = comparison_company_df["Percentage of Supported Claims"] - comparison_company_df["Percentage of Contradicted Claims"]

comp_colour = {"Mature": "green", "Moderate Maturity": "orange", "Least Mature": "purple"}
comp_supp_axis_entries = []
comp_contra_axis_entries = []
comp_ver_axis_entries = []
comp_NEI_axis_entries = []
comp_scatter_plot = plt.figure(figsize = (25, 20))
comp_scatter_veri_axis = comp_scatter_plot.add_subplot(2, 2, 1)
comp_scatter_veri_axis.set_xlabel("Total Labels")
comp_scatter_veri_axis.set_ylabel("Percentage of Verifiability")
comp_scatter_NEI_axis = comp_scatter_plot.add_subplot(2, 2, 2)
comp_scatter_NEI_axis.set_xlabel("Total Labels")
comp_scatter_NEI_axis.set_ylabel("Percentage of NEI Claims")
comp_scatter_supp_axis = comp_scatter_plot.add_subplot(2, 2, 3)
comp_scatter_supp_axis.set_xlabel("Total Labels")
comp_scatter_supp_axis.set_ylabel("Percentage of Supported Claims")
comp_scatter_contra_axis = comp_scatter_plot.add_subplot(2, 2, 4)
comp_scatter_contra_axis.set_xlabel("Total Labels")
comp_scatter_contra_axis.set_ylabel("Percentage of Contradicted Claims")
comp_scatter_veri_axis.set_xscale("log")
comp_scatter_NEI_axis.set_xscale("log")
comp_scatter_supp_axis.set_xscale("log")
comp_scatter_contra_axis.set_xscale("log")
comp_scatter_veri_axis.axhline(y = 0, color = "red", lw = 1)
comp_scatter_NEI_axis.axhline(y = 0, color = "red", lw = 1)
comp_scatter_supp_axis.axhline(y = 0, color = "red", lw = 1)
comp_scatter_contra_axis.axhline(y = 0, color = "red", lw = 1)

for index, row in comparison_company_df.iterrows(): 
    color = comp_colour[comparison_company[row["company"]]]
    comp_scatter_supp_axis.scatter(row["Total Labels"], row["Percentage of Supported Claims"], c = color)
    comp_scatter_contra_axis.scatter(row["Total Labels"], row["Percentage of Contradicted Claims"], c = color)
    comp_scatter_veri_axis.scatter(row["Total Labels"], row["Percentage of Verifiability"], c = color)
    comp_scatter_NEI_axis.scatter(row["Total Labels"], row["Percentage of NEI Claims"], c = color)
    # turning the three data points x, y, company name into text obj, to add to list for adjust text to work
    comp_ver_axis_entries.append(comp_scatter_veri_axis.text(row["Total Labels"], row["Percentage of Verifiability"], row["company"], fontsize = 15))
    comp_NEI_axis_entries.append(comp_scatter_NEI_axis.text(row["Total Labels"], row["Percentage of NEI Claims"], row["company"], fontsize = 15))
    comp_supp_axis_entries.append(comp_scatter_supp_axis.text(row["Total Labels"], row["Percentage of Supported Claims"], row["company"], fontsize = 15))
    comp_contra_axis_entries.append(comp_scatter_contra_axis.text(row["Total Labels"], row["Percentage of Contradicted Claims"], row["company"], fontsize = 15))

adjust_text(comp_ver_axis_entries, ax = comp_scatter_veri_axis, arrowprops = dict(arrowstyle = "-", color = "grey", lw = 0.5))
adjust_text(comp_NEI_axis_entries, ax = comp_scatter_NEI_axis, arrowprops = dict(arrowstyle = "-", color = "grey", lw = 0.5))
adjust_text(comp_supp_axis_entries, ax = comp_scatter_supp_axis, arrowprops = dict(arrowstyle = "-", color = "grey", lw = 0.5))
adjust_text(comp_contra_axis_entries, ax = comp_scatter_contra_axis, arrowprops = dict(arrowstyle = "-", color = "grey", lw = 0.5))
plt.savefig("comparison_companies_scatter_log_v2.png")

# No log
comp_supp_axis_entries = []
comp_contra_axis_entries = []
comp_ver_axis_entries = []
comp_NEI_axis_entries = []
comp_scatter_plot = plt.figure(figsize = (25, 20))
comp_scatter_veri_axis = comp_scatter_plot.add_subplot(2, 2, 1)
comp_scatter_veri_axis.set_xlabel("Total Labels")
comp_scatter_veri_axis.set_ylabel("Percentage of Verifiability")
comp_scatter_NEI_axis = comp_scatter_plot.add_subplot(2, 2, 2)
comp_scatter_NEI_axis.set_xlabel("Total Labels")
comp_scatter_NEI_axis.set_ylabel("Percentage of NEI Claims")
comp_scatter_supp_axis = comp_scatter_plot.add_subplot(2, 2, 3)
comp_scatter_supp_axis.set_xlabel("Total Labels")
comp_scatter_supp_axis.set_ylabel("Percentage of Supported Claims")
comp_scatter_contra_axis = comp_scatter_plot.add_subplot(2, 2, 4)
comp_scatter_contra_axis.set_xlabel("Total Labels")
comp_scatter_contra_axis.set_ylabel("Percentage of Contradicted Claims")
comp_scatter_veri_axis.axhline(y = 0, color = "red", lw = 1)
comp_scatter_NEI_axis.axhline(y = 0, color = "red", lw = 1)
comp_scatter_supp_axis.axhline(y = 0, color = "red", lw = 1)
comp_scatter_contra_axis.axhline(y = 0, color = "red", lw = 1)

for index, row in comparison_company_df.iterrows(): 
    color = comp_colour[comparison_company[row["company"]]]
    comp_scatter_supp_axis.scatter(row["Total Labels"], row["Percentage of Supported Claims"], c = color)
    comp_scatter_contra_axis.scatter(row["Total Labels"], row["Percentage of Contradicted Claims"], c = color)
    comp_scatter_veri_axis.scatter(row["Total Labels"], row["Percentage of Verifiability"], c = color)
    comp_scatter_NEI_axis.scatter(row["Total Labels"], row["Percentage of NEI Claims"], c = color)
    comp_ver_axis_entries.append(comp_scatter_veri_axis.text(row["Total Labels"], row["Percentage of Verifiability"], row["company"], fontsize = 10))
    comp_NEI_axis_entries.append(comp_scatter_NEI_axis.text(row["Total Labels"], row["Percentage of NEI Claims"], row["company"], fontsize = 10))
    comp_supp_axis_entries.append(comp_scatter_supp_axis.text(row["Total Labels"], row["Percentage of Supported Claims"], row["company"], fontsize = 10))
    comp_contra_axis_entries.append(comp_scatter_contra_axis.text(row["Total Labels"], row["Percentage of Contradicted Claims"], row["company"], fontsize = 10))

adjust_text(comp_ver_axis_entries, ax = comp_scatter_veri_axis, arrowprops = dict(arrowstyle = "-", color = "grey", lw = 0.5))
adjust_text(comp_NEI_axis_entries, ax = comp_scatter_NEI_axis, arrowprops = dict(arrowstyle = "-", color = "grey", lw = 0.5))
adjust_text(comp_supp_axis_entries, ax = comp_scatter_supp_axis, arrowprops = dict(arrowstyle = "-", color = "grey", lw = 0.5))
adjust_text(comp_contra_axis_entries, ax = comp_scatter_contra_axis, arrowprops = dict(arrowstyle = "-", color = "grey", lw = 0.5))
plt.savefig("comparison_companies_scatter_v2.png")

## This is for the comparison companies separated based on publication groups
comparison_company_df_by_group = company_df_by_group[company_df_by_group["company"].isin(comparison_company.keys())]
comparison_company_df_by_group["Percentage of Supported Claims"] = (comparison_company_df_by_group["SUPPORT"].fillna(0) / comparison_company_df_by_group["Total Labels"]) * 100
comparison_company_df_by_group["Percentage of Contradicted Claims"] = (comparison_company_df_by_group["CONTRADICT"].fillna(0) / comparison_company_df_by_group["Total Labels"]) * 100
comparison_company_df_by_group["Percentage of NEI Claims"] = (comparison_company_df_by_group["NEI"].fillna(0) / comparison_company_df_by_group["Total Labels"]) * 100
comparison_company_df_by_group["Percentage of Verifiability"] = comparison_company_df_by_group["Percentage of Supported Claims"] - comparison_company_df_by_group["Percentage of Contradicted Claims"]

# This dataframe to compare total percentage of supp/contra/nei for all comparison companies for both groups
group_stats_breakdown = company_df_by_group[company_df_by_group["company"].isin(comparison_company.keys())]
group_stats_breakdown["SUPPORT"] = comparison_company_df_by_group["SUPPORT"].fillna(0)
group_stats_breakdown["CONTRADICT"] = comparison_company_df_by_group["CONTRADICT"].fillna(0)
group_stats_breakdown["NEI"] = comparison_company_df_by_group["NEI"].fillna(0)
# total_stats_breakdown is the total claim break down for each company both pub groups combined
total_stats_breakdown = group_stats_breakdown.groupby("company")[["SUPPORT", "CONTRADICT", "NEI"]].sum()
# total_group_stats is total claim breakdown for each group, all combines combined
total_group_stats = group_stats_breakdown.groupby("Publication Group")[["SUPPORT", "CONTRADICT", "NEI"]].sum()
print(total_group_stats)
print(total_stats_breakdown)
print(group_stats_breakdown)

comp_colour = {"Mature": "green", "Moderate Maturity": "orange", "Least Mature": "purple"}
# Plot 4 subplots for each group
for group in ["Group_Nuclear_Publications", "Group_Reddit"]:
    print(f"For {group} Group:")
    group_df = comparison_company_df_by_group[comparison_company_df_by_group["Publication Group"] == group]

    # Plotting with log(x)
    comp_supp_axis_entries = []
    comp_contra_axis_entries = []
    comp_ver_axis_entries = []
    comp_NEI_axis_entries = []
    comp_scatter_plot = plt.figure(figsize = (25, 20))
    comp_scatter_veri_axis = comp_scatter_plot.add_subplot(2, 2, 1)
    comp_scatter_veri_axis.set_xlabel("Total Labels")
    comp_scatter_veri_axis.set_ylabel("Percentage of Verifiability")
    comp_scatter_NEI_axis = comp_scatter_plot.add_subplot(2, 2, 2)
    comp_scatter_NEI_axis.set_xlabel("Total Labels")
    comp_scatter_NEI_axis.set_ylabel("Percentage of NEI Claims")
    comp_scatter_supp_axis = comp_scatter_plot.add_subplot(2, 2, 3)
    comp_scatter_supp_axis.set_xlabel("Total Labels")
    comp_scatter_supp_axis.set_ylabel("Percentage of Supported Claims")
    comp_scatter_contra_axis = comp_scatter_plot.add_subplot(2, 2, 4)
    comp_scatter_contra_axis.set_xlabel("Total Labels")
    comp_scatter_contra_axis.set_ylabel("Percentage of Contradicted Claims")
    comp_scatter_veri_axis.set_xscale("log")
    comp_scatter_NEI_axis.set_xscale("log")
    comp_scatter_supp_axis.set_xscale("log")
    comp_scatter_contra_axis.set_xscale("log")
    comp_scatter_veri_axis.axhline(y = 0, color = "red", lw = 1) # demarcate the zero percent verifiability
    comp_scatter_NEI_axis.axhline(y = 0, color = "red", lw = 1)
    comp_scatter_supp_axis.axhline(y = 0, color = "red", lw = 1)
    comp_scatter_contra_axis.axhline(y = 0, color = "red", lw = 1)

    for index, row in group_df.iterrows(): # plotting percentage of supp/contra/nei/veri for each company against its log(No of claims) for each pub group
        color = comp_colour[comparison_company[row["company"]]]
        comp_scatter_supp_axis.scatter(row["Total Labels"], row["Percentage of Supported Claims"], c = color)
        comp_scatter_contra_axis.scatter(row["Total Labels"], row["Percentage of Contradicted Claims"], c = color)
        comp_scatter_veri_axis.scatter(row["Total Labels"], row["Percentage of Verifiability"], c = color)
        comp_scatter_NEI_axis.scatter(row["Total Labels"], row["Percentage of NEI Claims"], c = color)
        comp_ver_axis_entries.append(comp_scatter_veri_axis.text(row["Total Labels"], row["Percentage of Verifiability"], row["company"], fontsize = 15))
        comp_NEI_axis_entries.append(comp_scatter_NEI_axis.text(row["Total Labels"], row["Percentage of NEI Claims"], row["company"], fontsize = 15))
        comp_supp_axis_entries.append(comp_scatter_supp_axis.text(row["Total Labels"], row["Percentage of Supported Claims"], row["company"], fontsize = 15))
        comp_contra_axis_entries.append(comp_scatter_contra_axis.text(row["Total Labels"], row["Percentage of Contradicted Claims"], row["company"], fontsize = 15))

    adjust_text(comp_ver_axis_entries, ax = comp_scatter_veri_axis, arrowprops = dict(arrowstyle = "-", color = "grey", lw = 0.5))
    adjust_text(comp_NEI_axis_entries, ax = comp_scatter_NEI_axis, arrowprops = dict(arrowstyle = "-", color = "grey", lw = 0.5))
    adjust_text(comp_supp_axis_entries, ax = comp_scatter_supp_axis, arrowprops = dict(arrowstyle = "-", color = "grey", lw = 0.5))
    adjust_text(comp_contra_axis_entries, ax = comp_scatter_contra_axis, arrowprops = dict(arrowstyle = "-", color = "grey", lw = 0.5))
    plt.savefig(f"comparison_companies_scatter_log_by_{group}_v2.png")

    # No log plot
    comp_supp_axis_entries = []
    comp_contra_axis_entries = []
    comp_ver_axis_entries = []
    comp_NEI_axis_entries = []
    comp_scatter_plot = plt.figure(figsize = (25, 20))
    comp_scatter_veri_axis = comp_scatter_plot.add_subplot(2, 2, 1)
    comp_scatter_veri_axis.set_xlabel("Total Labels")
    comp_scatter_veri_axis.set_ylabel("Percentage of Verifiability")
    comp_scatter_NEI_axis = comp_scatter_plot.add_subplot(2, 2, 2)
    comp_scatter_NEI_axis.set_xlabel("Total Labels")
    comp_scatter_NEI_axis.set_ylabel("Percentage of NEI Claims")
    comp_scatter_supp_axis = comp_scatter_plot.add_subplot(2, 2, 3)
    comp_scatter_supp_axis.set_xlabel("Total Labels")
    comp_scatter_supp_axis.set_ylabel("Percentage of Supported Claims")
    comp_scatter_contra_axis = comp_scatter_plot.add_subplot(2, 2, 4)
    comp_scatter_contra_axis.set_xlabel("Total Labels")
    comp_scatter_contra_axis.set_ylabel("Percentage of Contradicted Claims")
    comp_scatter_veri_axis.axhline(y = 0, color = "red", lw = 1)
    comp_scatter_NEI_axis.axhline(y = 0, color = "red", lw = 1)
    comp_scatter_supp_axis.axhline(y = 0, color = "red", lw = 1)
    comp_scatter_contra_axis.axhline(y = 0, color = "red", lw = 1)

    for index, row in group_df.iterrows(): # plotting percentage of supp/contra/nei/veri for each company against its No of claims for each pub group
        color = comp_colour[comparison_company[row["company"]]]
        comp_scatter_supp_axis.scatter(row["Total Labels"], row["Percentage of Supported Claims"], c = color)
        comp_scatter_contra_axis.scatter(row["Total Labels"], row["Percentage of Contradicted Claims"], c = color)
        comp_scatter_veri_axis.scatter(row["Total Labels"], row["Percentage of Verifiability"], c = color)
        comp_scatter_NEI_axis.scatter(row["Total Labels"], row["Percentage of NEI Claims"], c = color)
        comp_ver_axis_entries.append(comp_scatter_veri_axis.text(row["Total Labels"], row["Percentage of Verifiability"], row["company"], fontsize = 15))
        comp_NEI_axis_entries.append(comp_scatter_NEI_axis.text(row["Total Labels"], row["Percentage of NEI Claims"], row["company"], fontsize = 15))
        comp_supp_axis_entries.append(comp_scatter_supp_axis.text(row["Total Labels"], row["Percentage of Supported Claims"], row["company"], fontsize = 15))
        comp_contra_axis_entries.append(comp_scatter_contra_axis.text(row["Total Labels"], row["Percentage of Contradicted Claims"], row["company"], fontsize = 15))

    adjust_text(comp_ver_axis_entries, ax = comp_scatter_veri_axis, arrowprops = dict(arrowstyle = "-", color = "grey", lw = 0.5))
    adjust_text(comp_NEI_axis_entries, ax = comp_scatter_NEI_axis, arrowprops = dict(arrowstyle = "-", color = "grey", lw = 0.5))
    adjust_text(comp_supp_axis_entries, ax = comp_scatter_supp_axis, arrowprops = dict(arrowstyle = "-", color = "grey", lw = 0.5))
    adjust_text(comp_contra_axis_entries, ax = comp_scatter_contra_axis, arrowprops = dict(arrowstyle = "-", color = "grey", lw = 0.5))
    plt.savefig(f"comparison_companies_scatter_by_{group}_v2.png")

    # 1.96 for 95% confidence interval
    confidence_z = 1.96 # for turning sigma into confidence interval, standardised for all companies
    corr_factor = 0.5

    # forest plot based on publication group
    forest_df_by_group = company_df_by_group[(company_df_by_group["company"].isin(comparison_company.keys())) & (company_df_by_group["Publication Group"] == group)]
    forest_df_by_group["Maturity"] = forest_df_by_group["company"].map(comparison_company)
    forest_df_by_group["Corrected Total Labels"] = forest_df_by_group["Total Labels"] + 3 * corr_factor
    forest_df_by_group["Percentage of Corrected Supported Claims"] = ((forest_df_by_group["SUPPORT"].fillna(0) + corr_factor) / forest_df_by_group["Corrected Total Labels"]) * 100
    forest_df_by_group["Percentage of Corrected Contradicted Claims"] = ((forest_df_by_group["CONTRADICT"].fillna(0) + corr_factor) / forest_df_by_group["Corrected Total Labels"]) * 100
    forest_df_by_group["Percentage of Corrected NEI Claims"] = ((forest_df_by_group["NEI"].fillna(0) + corr_factor) / forest_df_by_group["Corrected Total Labels"]) * 100
    forest_df_by_group["Percentage of Corrected Verifiability"] = forest_df_by_group["Percentage of Corrected Supported Claims"] - forest_df_by_group["Percentage of Corrected Contradicted Claims"]
    # convert variance to SE
    forest_df_by_group["Standard Error"] = np.sqrt((forest_df_by_group["Percentage of Corrected Supported Claims"] / 100 + forest_df_by_group["Percentage of Corrected Contradicted Claims"] / 100 - (forest_df_by_group["Percentage of Corrected Supported Claims"] / 100 - forest_df_by_group["Percentage of Corrected Contradicted Claims"] / 100) ** 2) / forest_df_by_group["Corrected Total Labels"]) * 100
    forest_df_by_group["Lower Limit"] = forest_df_by_group["Percentage of Corrected Verifiability"] - confidence_z * forest_df_by_group["Standard Error"]
    forest_df_by_group["Higher Limit"] = forest_df_by_group["Percentage of Corrected Verifiability"] + confidence_z * forest_df_by_group["Standard Error"]
    forest_df_by_group["Claim Count"] = forest_df_by_group["Total Labels"].astype(int).astype(str) # float to int to str, to remove trailing zeros
    
    ## Checking whether the verifiability results for each company is statistically significant
    # calculate for each company what is their own z score for BC test
    forest_df_by_group["Z Score"] = forest_df_by_group["Percentage of Corrected Verifiability"] / forest_df_by_group["Standard Error"]
    # cdf returns prob of value being less than or equal to z score x Sigma
    forest_df_by_group["P Value"] = 2 * (1 - stats.norm.cdf(forest_df_by_group["Z Score"].abs())) # two tailed

    ## Apply Bonferroni Correction
    rejected, p_corrected, _, _ = multipletests(forest_df_by_group["P Value"], alpha = 0.05, method = "bonferroni")
    forest_df_by_group["P Value Bonferroni Corrected"] = p_corrected
    forest_df_by_group["Significant after Bonferroni Correction"] = rejected
    # statistically sig companies with raw p values less than 0.05
    significant_companies = forest_df_by_group[forest_df_by_group["P Value"] < 0.05]
    # sig companies that survive correction
    significant_companies_corr = forest_df_by_group[forest_df_by_group["P Value Bonferroni Corrected"] < 0.05]

    # companies with stat significance based on raw p values
    for _, entry in significant_companies.iterrows():
        company = entry["company"]
        corrected_p = entry["P Value"]
        print(f"{company} is statistically significant before Bonferroni Correction, with raw p value of {corrected_p:.4f} (less than 0.05)")
    # for those that survive the correction
    for _, entry in significant_companies_corr.iterrows():
        company = entry["company"]
        corrected_p = entry["P Value Bonferroni Corrected"]
        print(f"{company} is significant after Bonferroni Correction, with p value of {corrected_p:.4f} (less than 0.05)")

    ## Spearman test to check relationship between number of labels and verifiability,
    correlation = stats.spearmanr(forest_df_by_group["Total Labels"], forest_df_by_group["Percentage of Corrected Verifiability"])
    corr_group = forest_df_by_group[forest_df_by_group["SUPPORT"].fillna(0) + forest_df_by_group["CONTRADICT"].fillna(0) > 0] # filter out companies with no support or contradict claims
    corr_correlation = stats.spearmanr(corr_group["Total Labels"], corr_group["Percentage of Corrected Verifiability"])

    print(f"(Including companies without Support or Contradict Labels) Spearman Correlation between Total Labels and Percentage of Corrected Verifiability: {correlation[0]:.4f}, p value is {correlation[1]:.4f}")
    print(f"(Excluding companies without Support or Contradict Labels) Spearman Correlation between Total Labels and Percentage of Corrected Verifiability: {corr_correlation[0]:.4f}, p value is {corr_correlation[1]:.4f}")

    fp.forestplot(forest_df_by_group,  
                estimate = "Percentage of Corrected Verifiability", 
                ll = "Lower Limit", hl = "Higher Limit",
                varlabel = "company",
                groupvar = "Maturity", 
                group_order = ["Mature", "Moderate Maturity", "Least Mature"],
                rightannote = ["Claim Count"],
                right_annoteheaders = ["N Claims"],
                xlabel = f"Percentage of Verifiability (95% Confidence Interval) for {group}",
                figsize = (6, 10)
                )

    plt.savefig(f"comparison_companies_forest_plot_by_{group}_v2.png", bbox_inches = "tight")

    # adding up all claims for each maturity group at company level, to find the average CI and % for each maturity group
    forest_df_by_group["Average CI Interval by Maturity"] = forest_df_by_group["Higher Limit"] - forest_df_by_group["Lower Limit"]
    mean_CI_by_maturity = forest_df_by_group.groupby("Maturity")["Average CI Interval by Maturity"].mean()
    total_labels_by_maturity = forest_df_by_group.groupby("Maturity")["Total Labels"].sum()
    # reindex it into wanted sequence: Mature, Moderate Maturity, Least Mature, previously auto sort by groupby, affected the sequence of maturity and the p score
    labels_by_maturity = forest_df_by_group.groupby("Maturity")[["SUPPORT", "CONTRADICT", "NEI"]].sum().reindex(["Mature", "Moderate Maturity", "Least Mature"])
    # div by matching based on row names
    percentage_by_maturity = (labels_by_maturity.div(total_labels_by_maturity, axis = 0)) * 100
    percentage_verifiability_maturity = percentage_by_maturity["SUPPORT"] - percentage_by_maturity["CONTRADICT"]

    # cochran armitage test for trend
    for label_type in ["SUPPORT", "CONTRADICT", "NEI"]:
        other_labels = [label for label in ["SUPPORT", "CONTRADICT", "NEI"] if label != label_type]
        # create table of claim counts, one col for counts for that label, the other for counts for the other labels
        support_table = np.column_stack([labels_by_maturity[label_type], labels_by_maturity[other_labels].sum(axis = 1)])
        support_trend = Table(support_table).test_ordinal_association()
        print(f"For {label_type}, Cochran Armitage test, z value is: {support_trend.zscore:.4f}, p value is {support_trend.pvalue:.4f}")
    # Average CI interval and percentage label for each maturity group
    for maturity in ["Mature", "Moderate Maturity", "Least Mature"]:
        print(f"For {maturity} group:")
        print(f"Average CI Interval is {mean_CI_by_maturity[maturity]:.2f}%")
        for label_type in ["SUPPORT", "CONTRADICT", "NEI"]:
            print(f"Percentage of {label_type} Claims is {percentage_by_maturity.loc[maturity, label_type]:.2f}%")
        print(f"Average Percentage Verifiability is {percentage_verifiability_maturity.loc[maturity]:.2f}%")
        print("\n")

    # For sorting underclaimed or overclaimed companies
    if group == "Group_Nuclear_Publications":
        filtered_group = forest_df_by_group[forest_df_by_group["SUPPORT"].fillna(0) + forest_df_by_group["CONTRADICT"].fillna(0) > 0] # filter out companies with no support and contra labels
        median_labels = filtered_group["Total Labels"].median()
        median_verifiability = filtered_group["Percentage of Corrected Verifiability"].median()
        # overclaim: above or equal to median label count with less than median corrected verifiability
        # underclaim: below median label count but more than or equal to median corrected verifiability
        overclaimed_companies = filtered_group[(filtered_group["Total Labels"] >= median_labels) & (filtered_group["Percentage of Corrected Verifiability"] < median_verifiability)]
        underclaimed_companies = filtered_group[(filtered_group["Total Labels"] < median_labels) & (filtered_group["Percentage of Corrected Verifiability"] >= median_verifiability)]

        print("Overclaimed Companies under Nuclear Publications Group include:")
        for _, entry in overclaimed_companies.iterrows():
            company = entry["company"]
            corrected_veri = entry["Percentage of Corrected Verifiability"]
            label = entry["Total Labels"]
            print(f"{company} is overclaimed with corrected verifiability of {corrected_veri:.2f}% and total labels of {label}")

        print("Underclaimed Companies under Nuclear Publications Group include:")
        for _, entry in underclaimed_companies.iterrows():
            company = entry["company"]
            corrected_veri = entry["Percentage of Corrected Verifiability"]
            label = entry["Total Labels"]
            print(f"{company} is underclaimed with corrected verifiability of {corrected_veri:.2f}% and total labels of {label}")

        
# forest plot
confidence_z = 1.96
# correction factor required for companies with zero support and contradict rates will have SE of zero
# leading to a more narrow confidence interval than other companies
corr_factor = 0.5

forest_df = company_df[company_df["company"].isin(comparison_company.keys())]
forest_df["Maturity"] = forest_df["company"].map(comparison_company)
forest_df["Corrected Total Labels"] = forest_df["Total Labels"] + 3 * corr_factor
forest_df["Percentage of Corrected Supported Claims"] = ((forest_df["SUPPORT"].fillna(0) + corr_factor ) / forest_df["Corrected Total Labels"]) * 100
forest_df["Percentage of Corrected Contradicted Claims"] = ((forest_df["CONTRADICT"].fillna(0) + corr_factor ) / forest_df["Corrected Total Labels"]) * 100
forest_df["Percentage of Corrected NEI Claims"] = ((forest_df["NEI"].fillna(0) + corr_factor ) / forest_df["Corrected Total Labels"]) * 100
forest_df["Percentage of Corrected Verifiability"] = forest_df["Percentage of Corrected Supported Claims"] - forest_df["Percentage of Corrected Contradicted Claims"]
# convert variance to SE
forest_df["Standard Error"] = np.sqrt((forest_df["Percentage of Corrected Supported Claims"] / 100 + forest_df["Percentage of Corrected Contradicted Claims"] / 100 - (forest_df["Percentage of Corrected Supported Claims"] / 100 - forest_df["Percentage of Corrected Contradicted Claims"] / 100) ** 2) / forest_df["Corrected Total Labels"]) * 100
forest_df["Lower Limit"] = forest_df["Percentage of Corrected Verifiability"] - confidence_z * forest_df["Standard Error"]
forest_df["Higher Limit"] = forest_df["Percentage of Corrected Verifiability"] + confidence_z * forest_df["Standard Error"]
forest_df["Claim Count"] = forest_df["Total Labels"].astype(int).astype(str)


fp.forestplot(forest_df,  
              estimate = "Percentage of Corrected Verifiability", 
              ll = "Lower Limit", hl = "Higher Limit",
              varlabel = "company",
              groupvar = "Maturity", 
              group_order = ["Mature", "Moderate Maturity", "Least Mature"],
              rightannote = ["Claim Count"],
              right_annoteheaders = ["N Claims"],
              xlabel = "Percentage of Verifiability (95% Confidence Interval)",
              figsize = (6, 10)
              )

plt.savefig("comparison_companies_forest_plot_v2.png", bbox_inches = "tight")

# comparison plot with and without reddit
grp_comp_supp_axis_entries = []
grp_comp_contra_axis_entries = []
grp_comp_ver_axis_entries = []
grp_comp_NEI_axis_entries = []
grp_comp_scatter_plot = plt.figure(figsize = (25, 20))
grp_comp_scatter_veri_axis = grp_comp_scatter_plot.add_subplot(2, 2, 1)
grp_comp_scatter_veri_axis.set_xlabel("Percentage of Verifiability (Total)")
grp_comp_scatter_veri_axis.set_ylabel("Percentage of Verifiability (Without Reddit)")
grp_comp_scatter_NEI_axis = grp_comp_scatter_plot.add_subplot(2, 2, 2)
grp_comp_scatter_NEI_axis.set_xlabel("Percentage of NEI Claims (Total)")
grp_comp_scatter_NEI_axis.set_ylabel("Percentage of NEI Claims (Without Reddit)")
# NEI plot start at around 70, points are too clustered togther, redefine starting axis limits
grp_comp_scatter_NEI_axis.set_ylim(70, 120)
grp_comp_scatter_NEI_axis.set_xlim(70, 120)
grp_comp_scatter_supp_axis = grp_comp_scatter_plot.add_subplot(2, 2, 3)
grp_comp_scatter_supp_axis.set_xlabel("Percentage of Supported Claims (Total)")
grp_comp_scatter_supp_axis.set_ylabel("Percentage of Supported Claims (Without Reddit)")
grp_comp_scatter_contra_axis = grp_comp_scatter_plot.add_subplot(2, 2, 4)
grp_comp_scatter_contra_axis.set_xlabel("Percentage of Contradicted Claims (Total)")
grp_comp_scatter_contra_axis.set_ylabel("Percentage of Contradicted Claims (Without Reddit)")
comparison_company_df_nuclear_pub = comparison_company_df_by_group[comparison_company_df_by_group["Publication Group"] == "Group_Nuclear_Publications"]
# merging with left df for companies that exist between both df, renaming conflicting header name by adding suffix for left and right df
comparison_company_df_merged = comparison_company_df_nuclear_pub.merge(comparison_company_df, on = "company", suffixes = (" (without Reddit)", " (Total)"))
for index, row in comparison_company_df_merged.iterrows():
    color = comp_colour[comparison_company[row["company"]]]
    grp_comp_scatter_supp_axis.scatter(row["Percentage of Supported Claims (Total)"], row["Percentage of Supported Claims (without Reddit)"], c = color)
    grp_comp_scatter_contra_axis.scatter(row["Percentage of Contradicted Claims (Total)"], row["Percentage of Contradicted Claims (without Reddit)"], c = color)
    grp_comp_scatter_veri_axis.scatter(row["Percentage of Verifiability (Total)"], row["Percentage of Verifiability (without Reddit)"], c = color)
    grp_comp_scatter_NEI_axis.scatter(row["Percentage of NEI Claims (Total)"], row["Percentage of NEI Claims (without Reddit)"], c = color)
    grp_comp_ver_axis_entries.append(grp_comp_scatter_veri_axis.text(row["Percentage of Verifiability (Total)"], row["Percentage of Verifiability (without Reddit)"], row["company"], fontsize = 15))
    grp_comp_NEI_axis_entries.append(grp_comp_scatter_NEI_axis.text(row["Percentage of NEI Claims (Total)"], row["Percentage of NEI Claims (without Reddit)"], row["company"], fontsize = 15))
    grp_comp_supp_axis_entries.append(grp_comp_scatter_supp_axis.text(row["Percentage of Supported Claims (Total)"], row["Percentage of Supported Claims (without Reddit)"], row["company"], fontsize = 15))
    grp_comp_contra_axis_entries.append(grp_comp_scatter_contra_axis.text(row["Percentage of Contradicted Claims (Total)"], row["Percentage of Contradicted Claims (without Reddit)"], row["company"], fontsize = 15))

# add a 45 deg line cutting origin, if value is above line means that removing reddit increased the percentage
for axis in [grp_comp_scatter_veri_axis, grp_comp_scatter_NEI_axis, grp_comp_scatter_supp_axis, grp_comp_scatter_contra_axis]:
    axis.axline((0, 0), slope = 1, color = "red", lw = 1)
adjust_text(grp_comp_ver_axis_entries, ax = grp_comp_scatter_veri_axis, expand = (2, 3), force_text = (2, 3), arrowprops = dict(arrowstyle = "-", color = "grey", lw = 0.5))
adjust_text(grp_comp_NEI_axis_entries, ax = grp_comp_scatter_NEI_axis, expand = (2, 3), force_text = (2, 3), arrowprops = dict(arrowstyle = "-", color = "grey", lw = 0.5))
adjust_text(grp_comp_supp_axis_entries, ax = grp_comp_scatter_supp_axis, expand = (2, 3), force_text = (2, 3), arrowprops = dict(arrowstyle = "-", color = "grey", lw = 0.5))
adjust_text(grp_comp_contra_axis_entries, ax = grp_comp_scatter_contra_axis, expand = (2, 3), force_text = (2, 3), arrowprops = dict(arrowstyle = "-", color = "grey", lw = 0.5))
plt.savefig("comparison_companies_scatter_with_without_reddit_v2.png")

# Data analysis, comparing the importance of each attribute in the chunk weight
query_claim_scores = pd.read_csv("query_claim_scores.csv")

# convert entire row which is a string, into a list of floats
relevance_scores = query_claim_scores["relevance_scores"].apply(ast.literal_eval)
stance_scores = query_claim_scores["stance_scores"].apply(ast.literal_eval)
provenance_scores = query_claim_scores["provenance_scores"].apply(ast.literal_eval)
recency_scores = query_claim_scores["recency_scores"].apply(ast.literal_eval)
source_cred_scores = query_claim_scores["source_credibility_scores"].apply(ast.literal_eval)
source_ids = query_claim_scores["relevant_source_ids"].apply(ast.literal_eval)

def calculate_score(entry_index, attribute_out = None, top_N = None):
    """
    Calculate query claim score, leave attribute out in calculation if specified, else calculated as per normal
    Have the option to vary the top N chunks retrieved and study its impact on query claim score
    """
    total_weight = 0
    total_score = 0
    if attribute_out == "recency":
        for i in range(len(stance_scores[entry_index])):
            provenance_score = provenance_scores[entry_index][i]
            relevance_score = relevance_scores[entry_index][i]
            source_cred_score = source_cred_scores[entry_index][i]
            chunk_weight = provenance_score * relevance_score * source_cred_score
            chunk_score = stance_scores[entry_index][i] * chunk_weight
            total_weight += chunk_weight
            total_score += chunk_score

    elif attribute_out == "relevance":
        for i in range(len(stance_scores[entry_index])):
            provenance_score = provenance_scores[entry_index][i]
            recency_score = recency_scores[entry_index][i]
            source_cred_score = source_cred_scores[entry_index][i]
            chunk_weight = provenance_score * recency_score * source_cred_score
            chunk_score = stance_scores[entry_index][i] * chunk_weight
            total_weight += chunk_weight
            total_score += chunk_score

    if attribute_out == "source credibility":
        for i in range(len(stance_scores[entry_index])):
            provenance_score = provenance_scores[entry_index][i]
            recency_score = recency_scores[entry_index][i]
            relevance_score = relevance_scores[entry_index][i]
            chunk_weight = provenance_score * recency_score * relevance_score
            chunk_score = stance_scores[entry_index][i] * chunk_weight
            total_weight += chunk_weight
            total_score += chunk_score
        
    if attribute_out == "provenance":
        for i in range(len(stance_scores[entry_index])):
            source_cred_score = source_cred_scores[entry_index][i]
            recency_score = recency_scores[entry_index][i]
            relevance_score = relevance_scores[entry_index][i]
            chunk_weight = source_cred_score * recency_score * relevance_score
            chunk_score = stance_scores[entry_index][i] * chunk_weight
            total_weight += chunk_weight
            total_score += chunk_score
    
    elif attribute_out == None and top_N is not None:
        top_N_chunks = 0
        existing_source_ids = set()

        # this is to consider chunks of different sources for top N
        for i in range(len(stance_scores[entry_index])):
            if top_N_chunks >= top_N:
                break # retrieved list of chunks is complete
            source_id = source_ids[entry_index][i]
            if source_id not in existing_source_ids: # for chunks of a diff source
                existing_source_ids.add(source_id)
                source_cred_score = source_cred_scores[entry_index][i]
                recency_score = recency_scores[entry_index][i]
                relevance_score = relevance_scores[entry_index][i]
                provenance_score = provenance_scores[entry_index][i]
                chunk_weight = source_cred_score * recency_score * relevance_score * provenance_score
                chunk_score = stance_scores[entry_index][i] * chunk_weight
                total_weight += chunk_weight
                total_score += chunk_score
                top_N_chunks += 1


    return total_score / total_weight if total_weight != 0 else 0

for attribute_out in ["relevance", "provenance", "recency", "source credibility"]:
    query_claim_score = []
    for entry_index in range(len(query_claim_scores)):
        query_claim_score.append(calculate_score(entry_index, attribute_out = attribute_out))

    query_claim_score_varying = pd.Series(query_claim_score).apply(classification, args = (support_threshold, contradict_threshold))
    label_counts = query_claim_score_varying.value_counts()
    print(f"For excluding {attribute_out} from query claim score calculation, the query claim label distribution as follows:")
    for label in ["SUPPORT", "CONTRADICT", "NEI"]:
        percentage = (label_counts[label] / len(query_claim_score)) * 100
        print(f"{label} Percentage: {percentage:.3f}%")

# for this run we try out different top N chunks retrieved and its impact on query claim scores
# we exclude chunks from the same source id in the top N
# we varied the IQR ranges using that from v2 and from its own data distribution
for top_N in [1, 3, 5, 7]:
    query_claim_score = []
    for entry_index in range(len(query_claim_scores)):
        query_claim_score.append(calculate_score(entry_index, attribute_out = None, top_N = top_N))
    # Classification with fixed threshold limits from v2 run vs using its own distribution IQR limits
    query_claim_score_varying = pd.Series(query_claim_score).apply(classification, args = (support_threshold, contradict_threshold)) # old fixed thresholds
    # with its own distribution IQR limits
    own_support_threshold, own_contradict_threshold = threshold_calculation(pd.Series(query_claim_score))
    query_claim_score_varying_own = pd.Series(query_claim_score).apply(classification, args = (own_support_threshold, own_contradict_threshold))
    label_counts = query_claim_score_varying.value_counts() # old IQR thresholds from v2 run
    label_counts_own = query_claim_score_varying_own.value_counts() # with own IQR thresholds
    print(f"For excluding chunks of the same source id in top {top_N}, the query claim label distribution as follows:")
    print("Using fixed thresholds from the earlier v2 run:")
    for label in ["SUPPORT", "CONTRADICT", "NEI"]:
        percentage = (label_counts.get(label, 0) / len(query_claim_score)) * 100
        print(f"{label} Percentage: {percentage:.3f}%")
    print("Using IQR thresholds from its own distribution:")
    for label in ["SUPPORT", "CONTRADICT", "NEI"]:
        percentage = (label_counts_own.get(label, 0) / len(query_claim_score)) * 100
        print(f"{label} Percentage: {percentage:.3f}%")
