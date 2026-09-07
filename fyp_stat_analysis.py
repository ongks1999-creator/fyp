import json
import re
import csv
from collections import defaultdict

# terms created by ChatGPT as documented in the report
terms = {
    "nuclear energy":[
    "nuclear reactor", "reactor", "fission reactor", "fusion reactor",
    "nuclear power", "nuclear power plant", "nuclear energy",
    "nuclear fuel", "fuel rod", "fuel rods", "fuel assembly",
    "control rod", "control rods", "moderator", "coolant",
    "reactor core", "core meltdown", "meltdown",
    "pressurized water reactor", "PWR", "boiling water reactor", "BWR",
    "fast breeder reactor", "breeder reactor", "small modular reactor", "SMR",
    "thermal reactor", "fast reactor", "nuclear facility",
    "containment vessel", "reactor vessel", "criticality",
    "subcritical", "supercritical", "chain reaction"],
    
    "nuclear reactions":[

    "nuclear fission", "fission", "nuclear fusion", "fusion",
    "radioactive decay", "decay", "alpha decay", "beta decay",
    "gamma decay", "electron capture", "neutron capture",
    "proton emission", "neutron emission", "spontaneous fission",
    "nuclear reaction", "transmutation", "activation",
    "binding energy", "mass defect", "half-life", "half life",
    "decay constant", "daughter isotope", "parent isotope",
    "decay series", "decay chain"],

    "nuclear radiation" : [
    "radiation", "ionising radiation", "ionizing radiation",
    "alpha radiation", "beta radiation", "gamma radiation",
    "neutron radiation", "x-ray", "x rays", "gamma ray",
    "gamma rays", "cosmic ray", "cosmic rays",
    "background radiation", "radioactivity", "radioactive",
    "irradiation", "radiation exposure", "radiation dose",
    "absorbed dose", "equivalent dose", "effective dose",
    "radiation poisoning", "radiation sickness",
    "radiation shielding", "shielding", "ionisation", "ionization"],

    "nuclear particles": [
    "neutron", "neutrons", "proton", "protons",
    "electron", "electrons", "positron", "positrons",
    "alpha particle", "alpha particles", "beta particle", "beta particles",
    "gamma photon", "photon", "photons", "nucleon", "nucleons",
    "nuclide", "nuclides", "isotope", "isotopes",
    "radioisotope", "radioisotopes", "radionuclide", "radionuclides"],

    "nuclear materials": [
    "uranium", "uranium-235", "uranium 235", "U-235", "U235",
    "uranium-238", "uranium 238", "U-238", "U238",
    "plutonium", "plutonium-239", "plutonium 239", "Pu-239", "Pu239",
    "thorium", "thorium-232", "Th-232", "radium", "radon",
    "polonium", "cesium", "caesium", "cesium-137", "caesium-137",
    "Cs-137", "strontium-90", "Sr-90", "iodine-131", "I-131",
    "tritium", "deuterium", "carbon-14", "C-14",
    "cobalt-60", "Co-60", "americium", "americium-241", "Am-241",
    "enriched uranium", "depleted uranium", "weapons-grade uranium",
    "highly enriched uranium", "low enriched uranium",
    "nuclear material", "radioactive material",
    "fissile material", "fertile material"],

    "nuclear weapons": [
    "nuclear weapon", "nuclear weapons", "atomic bomb", "hydrogen bomb",
    "thermonuclear weapon", "nuclear warhead", "warhead",
    "nuclear missile", "ballistic missile", "ICBM",
    "nuclear test", "nuclear testing", "nuclear explosion",
    "nuclear blast", "fallout", "radioactive fallout",
    "nuclear proliferation", "non-proliferation",
    "nuclear deterrence", "nuclear arsenal",
    "nuclear disarmament", "nuclear arms", "nuclear arms race",
    "critical mass", "dirty bomb", "radiological weapon"],

    "nuclear waste": [
    "nuclear waste", "radioactive waste", "spent fuel",
    "spent nuclear fuel", "high-level waste", "low-level waste",
    "intermediate-level waste", "waste repository",
    "geological repository", "deep geological repository",
    "radioactive contamination", "contamination", "decontamination",
    "nuclear accident", "radiological accident", "reactor accident",
    "Chernobyl", "Fukushima", "Three Mile Island",
    "nuclear safety", "radiation safety", "radiation protection",
    "dose limit", "exclusion zone", "evacuation zone"],

    "general physics": [
    "physics", "mechanics", "classical mechanics", "motion",
    "force", "forces", "mass", "acceleration", "velocity",
    "speed", "displacement", "distance", "momentum", "impulse",
    "inertia", "torque", "angular momentum", "work", "energy",
    "kinetic energy", "potential energy", "mechanical energy",
    "power", "friction", "gravity", "gravitation",
    "gravitational force", "weight", "free fall",
    "projectile motion", "Newton's laws", "Newtonian mechanics",
    "equilibrium", "centre of mass", "center of mass"],

    "physics waves": [
    "wave", "waves", "oscillation", "oscillations",
    "vibration", "frequency", "wavelength", "amplitude",
    "period", "phase", "wave speed", "interference",
    "diffraction", "reflection", "refraction", "standing wave",
    "resonance", "harmonic motion", "simple harmonic motion",
    "sound wave", "acoustic wave", "ultrasound", "Doppler effect"],

    "physics electricity": [
    "electricity", "electric field", "electric charge",
    "charge", "charges", "current", "electric current",
    "voltage", "potential difference", "resistance",
    "resistor", "capacitance", "capacitor", "inductance",
    "inductor", "circuit", "circuits", "Ohm's law",
    "Coulomb's law", "magnetism", "magnetic field",
    "electromagnetism", "electromagnetic force",
    "electromagnetic wave", "electromagnetic radiation",
    "Faraday's law", "Ampere's law", "Maxwell's equations",
    "Lorentz force", "flux", "magnetic flux"],

    "physics thermodynamics": [
    "thermodynamics", "temperature", "heat", "thermal energy",
    "internal energy", "entropy", "enthalpy", "specific heat",
    "heat capacity", "latent heat", "conduction", "convection",
    "thermal radiation", "thermal equilibrium", "absolute zero",
    "Kelvin", "pressure", "volume", "ideal gas", "gas law",
    "Boyle's law", "Charles's law",
    "first law of thermodynamics", "second law of thermodynamics",
    "heat engine", "Carnot cycle"],

    "physics optics": [
    "light", "optics", "ray", "rays", "lens", "lenses",
    "mirror", "mirrors", "polarisation", "polarization",
    "dispersion", "prism", "laser", "visible light",
    "infrared", "ultraviolet", "UV", "electromagnetic spectrum",
    "refractive index", "focal length", "image formation"],

    "quantum physics": [
    "quantum", "quantum physics", "quantum mechanics",
    "quantum theory", "wavefunction", "wave function",
    "Schrodinger equation", "Schrödinger equation",
    "uncertainty principle", "Heisenberg uncertainty principle",
    "superposition", "quantum state", "energy level",
    "quantisation", "quantization", "wave-particle duality",
    "particle-wave duality", "photoelectric effect",
    "Planck constant", "Planck's constant",
    "quantum tunnelling", "quantum tunneling", "spin",
    "quantum number", "orbital", "qubit", "entanglement",
    "quantum entanglement", "particle physics",
    "elementary particle", "subatomic particle", "standard model",
    "quark", "quarks", "lepton", "leptons",
    "muon", "tau", "neutrino", "neutrinos",
    "boson", "bosons", "fermion", "fermions",
    "Higgs boson", "gluon", "W boson", "Z boson",
    "antimatter", "antiparticle", "hadron", "hadrons",
    "baryon", "baryons", "meson", "mesons",
    "particle accelerator", "collider", "Large Hadron Collider",
    "LHC", "CERN"],

    "astrophysics": [
    "relativity", "special relativity", "general relativity",
    "Einstein", "spacetime", "space-time", "time dilation",
    "length contraction", "mass-energy equivalence",
    "E=mc2", "E = mc^2", "speed of light",
    "black hole", "black holes", "event horizon",
    "gravitational wave", "gravitational waves",
    "cosmology", "universe", "galaxy", "galaxies",
    "star", "stars", "stellar", "supernova",
    "neutron star", "white dwarf", "dark matter",
    "dark energy", "big bang", "redshift",
    "cosmic microwave background", "CMB"],

    "physics SI units": [
    "joule", "joules", "newton", "newtons", "watt", "watts",
    "pascal", "pascals", "volt", "volts", "ampere",
    "amperes", "ohm", "ohms", "tesla", "weber",
    "coulomb", "coulombs", "farad", "henry", "hertz",
    "becquerel", "gray", "sievert", "curie",
    "electronvolt", "electron volt", "eV", "MeV", "GeV",
    "kelvin", "mole", "mol", "m/s", "m/s^2"]
}
def load_json(file_path): # used for fever dataset
    fever = []
    with open(file_path,'r') as file:
        for line in file:
            fever.append(json.loads(line))

    return fever

def compile_terms(terms):
    compiled_terms = {}
    for category, keywords in terms.items():
        # join words with word boundary for exact keyword match, turn special characters into literal string, compile all into one regex pattern
        compiled_terms[category] = re.compile('|'.join(r"\b" + re.escape(keyword.lower()) + r'\b' for keyword in keywords))

    return compiled_terms

compiled_terms = compile_terms(terms)

def analyse_dataset(compiled_terms, claim, count):
    for category, pattern in compiled_terms.items(): # one compiled pattern per category
        
        if pattern.search(claim): # search within that claim for any keyword in the compiled term list, if there is at least one keyword match, it counts as one
                count[category] += 1

def data_analysis(results, data_set_length):
    """
    Each claim matched based on the sub categories within the category (nuclear/physics)
    If one has multiple sub-category matches, each of the matched sub-cat is counted once
    """
    physics_total = 0
    nuclear_total = 0
    for category, count in results.items():
        if "physics" in category:
            physics_total += count
        elif "nuclear" in category:
            nuclear_total += count

    print(f"Total physics related claims: {physics_total}")
    print(f"Total physics related claims (%): {physics_total / data_set_length:.2%}")
    print(f"Total nuclear related claims: {nuclear_total}")
    print(f"Total nuclear related claims (%): {nuclear_total / data_set_length:.2%}")


def load_csv(file_path):
    data = {}
    with open(file_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data[row["doc_id"]] = (row["title"] or "") + (row["abstract"] or "")
    return data

"""
fever_data = load_json("/Users/ongkaisheng/Desktop/ImperialCollege/FYP/fever/train.jsonl")
fever_claims_set = set()
# mutation rule in fever might produce a duplicate claim text (https://arxiv.org/pdf/1803.05355), use set to keep track
fever_count = defaultdict(int) # default value of zero for each category
for item in fever_data:
    claim = item["claim"].lower()
    if claim in fever_claims_set:
        continue
    fever_claims_set.add(claim) # verify if claim has been seen since there is possible duplication
    analyse_dataset(compiled_terms, claim, fever_count)

print(len(fever_claims_set))
print(fever_count)
data_analysis(fever_count, len(fever_claims_set))
"""




scifact_data = load_csv("/Users/ongkaisheng/Desktop/ImperialCollege/FYP/fyp/scifact1/corpus_train.csv")
# Sci fact data analysis
scifact_count = defaultdict(int)
# since each key being a unique id, if there are duplicates it overwrites with no duplicated keys
# while fever has same text but different id so if there is duplication the same claim counts extra
for id, content in scifact_data.items():
    analyse_dataset(compiled_terms, content.lower(), scifact_count)
print(len(scifact_data))
print(scifact_count)
data_analysis(scifact_count, len(scifact_data))



"""
dbpedia_count = defaultdict(int)
dbpedia_file_length = 4635922 # unique docs based on what was documented on huggingface
with open("/Users/ongkaisheng/Desktop/ImperialCollege/FYP/dbpedia/corpus.jsonl", "r") as file:
        for index, line in enumerate(file):
            line = json.loads(line)
            entry_data = (line["title"].lower() or "") + (line["text"].lower() or "")
            analyse_dataset(compiled_terms, entry_data, dbpedia_count)

            if index > 0 and index % (dbpedia_file_length // 10) == 0:
                # loads progress every 10%
                print(f"Loaded {index} of total file")

print(dbpedia_count)
data_analysis(dbpedia_count, dbpedia_file_length)
"""

