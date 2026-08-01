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
label_counts = results_df["Verdict Label"].value_counts()
# Label Statistical Breakdown
for label in ["SUPPORT", "CONTRADICT", "NEI"]:
    percentage = (label_counts[label] / len(results_df)) * 100
    print(f"{label} Percentage: {percentage:.3f}%")

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
               unicodedata.normalize("NFC", "USNC"): "Ultra Safe Nuclear Corporation",
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
               unicodedata.normalize("NFC", "UltraSafe Nuclear"): "Ultra Safe Nuclear Corporation",
               unicodedata.normalize("NFC", "Ultra Safe Nuclear Corp (USNC)"): "Ultra Safe Nuclear Corporation",
               unicodedata.normalize("NFC", "Ultra Safe Nuclear"): "Ultra Safe Nuclear Corporation",
               unicodedata.normalize("NFC", "ThorCon International"): "ThorCon",
               unicodedata.normalize("NFC", "Thorcon"): "ThorCon",
               unicodedata.normalize("NFC", "General Atomics Electromagnetic Systems (GA-EMS)"): "General Atomics",
               unicodedata.normalize("NFC", "Yellowcake PLC"): "Yellow Cake",
               unicodedata.normalize("NFC", "ČEZ"): "CEZ",
               unicodedata.normalize("NFC", "ORNL"): "Oak Ridge National Laboratory (ORNL)",
               unicodedata.normalize("NFC", "CNL"): "Canadian Nuclear Laboratories (CNL)"}
# normalise it for unicode, same as mapping_dict keys

for company, standardised_name in mapping_dict.items():
    if standardised_name in alias_names.keys():
        mapping_dict[company] = alias_names[standardised_name]

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
comparison_company = {"Framatome": ("Framatome", "Mature"),
                      "Westinghouse": ("Westinghouse", "Mature"),
                     "NuScale": ("NuScale", "Moderate Maturity"),
                     "Holtec": ("Holtec International", "Moderate Maturity"),
                     "GE-Hitachi": ("GE Hitachi", "Moderate Maturity"),
                     "ARC": ("ARC", "Moderate Maturity"),
                     "Moltex": ("Moltex Energy", "Moderate Maturity"),
                     "ThorCon": ("ThorCon", "Least Mature"),
                     "Ultra Safe Nuclear Corporation": ("Ultra Safe Nuclear Corporation", "Least Mature"),
                     "General Atomics": ("General Atomics", "Least Mature"),
                     "Flibe": ("Flibe", "Least Mature"),
                     "NANO": ("NANO Nuclear", "Least Mature"),
                     "Oklo": ("Oklo", "Least Mature"),
                     "Kairos": ("Kairos Power", "Least Mature"),
                     "BWXT": ("BWX", "Least Mature"),
                     "X-Energy": ("X-Energy", "Least Mature"),
                     "Oak Ridge National Laboratory (ORNL)": ("Oak Ridge National Laboratory", "Least Mature"),
                     "TerraPower": ("TerraPower", "Least Mature")
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

for index, row in comparison_company_df.iterrows(): 
    color = comp_colour[comparison_company[row["company"]][1]]
    comp_scatter_supp_axis.scatter(row["Total Labels"], row["Percentage of Supported Claims"], c = color)
    comp_scatter_contra_axis.scatter(row["Total Labels"], row["Percentage of Contradicted Claims"], c = color)
    comp_scatter_veri_axis.scatter(row["Total Labels"], row["Percentage of Verifiability"], c = color)
    comp_scatter_NEI_axis.scatter(row["Total Labels"], row["Percentage of NEI Claims"], c = color)
    comp_ver_axis_entries.append(comp_scatter_veri_axis.text(row["Total Labels"], row["Percentage of Verifiability"], row["company"], fontsize = 15))
    comp_NEI_axis_entries.append(comp_scatter_NEI_axis.text(row["Total Labels"], row["Percentage of NEI Claims"], row["company"], fontsize = 15))
    comp_supp_axis_entries.append(comp_scatter_supp_axis.text(row["Total Labels"], row["Percentage of Supported Claims"], row["company"], fontsize = 15))
    comp_contra_axis_entries.append(comp_scatter_contra_axis.text(row["Total Labels"], row["Percentage of Contradicted Claims"], row["company"], fontsize = 15))

adjust_text(comp_ver_axis_entries, ax = comp_scatter_veri_axis)
adjust_text(comp_NEI_axis_entries, ax = comp_scatter_NEI_axis)
adjust_text(comp_supp_axis_entries, ax = comp_scatter_supp_axis)
adjust_text(comp_contra_axis_entries, ax = comp_scatter_contra_axis)
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

for index, row in comparison_company_df.iterrows(): 
    color = comp_colour[comparison_company[row["company"]][1]]
    comp_scatter_supp_axis.scatter(row["Total Labels"], row["Percentage of Supported Claims"], c = color)
    comp_scatter_contra_axis.scatter(row["Total Labels"], row["Percentage of Contradicted Claims"], c = color)
    comp_scatter_veri_axis.scatter(row["Total Labels"], row["Percentage of Verifiability"], c = color)
    comp_scatter_NEI_axis.scatter(row["Total Labels"], row["Percentage of NEI Claims"], c = color)
    comp_ver_axis_entries.append(comp_scatter_veri_axis.text(row["Total Labels"], row["Percentage of Verifiability"], row["company"], fontsize = 15))
    comp_NEI_axis_entries.append(comp_scatter_NEI_axis.text(row["Total Labels"], row["Percentage of NEI Claims"], row["company"], fontsize = 15))
    comp_supp_axis_entries.append(comp_scatter_supp_axis.text(row["Total Labels"], row["Percentage of Supported Claims"], row["company"], fontsize = 15))
    comp_contra_axis_entries.append(comp_scatter_contra_axis.text(row["Total Labels"], row["Percentage of Contradicted Claims"], row["company"], fontsize = 15))

adjust_text(comp_ver_axis_entries, ax = comp_scatter_veri_axis)
adjust_text(comp_NEI_axis_entries, ax = comp_scatter_NEI_axis)
adjust_text(comp_supp_axis_entries, ax = comp_scatter_supp_axis)
adjust_text(comp_contra_axis_entries, ax = comp_scatter_contra_axis)
plt.savefig("comparison_companies_scatter_v2.png")

# forest plot
confidence_z = 1.96
# correction factor required for companies with zero support and contradict rates will have SE of zero
# leading to a more narrow confidence interval than other companies
corr_factor = 0.5

forest_df = company_df[company_df["company"].isin(comparison_company.keys())].copy()
comparison_company = {company: maturity for company, (name, maturity) in comparison_company.items()}
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