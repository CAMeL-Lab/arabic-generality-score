# distance_function/substitution_weight.py
# Load the probability tables written by precompute_probabilities.py and expose
# compute_substitution_cost(): a phonology-aware cost for one character substitution.

import os
import json
import pandas as pd

# --------- config / paths ----------
OUTPUT_DIR = os.environ.get("AGS_OUTPUT_DIR", "output")
CAPHI_TABLE = os.environ.get("AGS_CAPHI_TABLE", "distance_function/data/CAPHI/CAPHI-table.tsv")

# --------- globals populated by load_distance_resources() ----------
caphi_coda_probability_df = None  # CAPHI_CODA_probabilities.tsv
raw_coda_probability_df = None    # RAW_CODA_probabilities.tsv
caphi_coda_dict = None            # from CAPHI_CODA_probabilities_flat.tsv
raw_coda_dict = None              # from RAW_CODA_probabilities_flat.tsv
dialect_dict = None               # from P_CAPHI_given_ORTHO_by_dialect.json
mappings = None                   # CAPHI letter->all CAPHI list (source letters universe)

# --------- helpers ----------
def _read_tsv(path):
    return pd.read_csv(path, sep="\t")

def _build_nested_from_flat(df, keys, value_cols):
    """
    Convert a flat DataFrame into a nested dict keyed by `keys`.
    value_cols retained as dict payload.
    """
    nested = {}
    for _, row in df.iterrows():
        k = tuple(row[k] for k in keys)
        if k not in nested:
            nested[k] = {}
        # Store as a single dict with the probability columns
        nested[k] = {c: row[c] for c in value_cols if c in row}
    return nested

def load_distance_resources(
    output_dir: str = OUTPUT_DIR,
    caphi_table_path: str = CAPHI_TABLE,
):
    """Populate globals from artifacts saved by compute_probabilities.py."""
    global caphi_coda_probability_df, raw_coda_probability_df
    global caphi_coda_dict, raw_coda_dict, dialect_dict, mappings

    # 1) Primary tidy probability tables
    caphi_coda_probability_df = _read_tsv(os.path.join(output_dir, "CAPHI_CODA_probabilities.tsv"))
    raw_coda_probability_df   = _read_tsv(os.path.join(output_dir, "RAW_CODA_probabilities.tsv"))

    # 2) Flat tables → dicts (preferred; contain default/phon/etym counts)
    caphi_flat_path = os.path.join(output_dir, "CAPHI_CODA_probabilities_flat.tsv")
    raw_flat_path   = os.path.join(output_dir, "RAW_CODA_probabilities_flat.tsv")

    if os.path.exists(caphi_flat_path):
        caphi_flat = _read_tsv(caphi_flat_path)
        caphi_coda_dict = {
            (row["Dialect"], row["CODA"], row["CAPHI"]): {
                "P(CODA|CAPHI)": row.get("P(CODA|CAPHI)", 0),
                "P(CAPHI|CODA)": row.get("P(CAPHI|CODA)", 0),
                "default mapping": bool(row.get("default_mapping", False)),
                "phon_count": int(row.get("phon_count", 0)),
                "etym_count": int(row.get("etym_count", 0)),
                # A persisted total of 0 means "never updated"; the thesis notebook had no
                # `total` key for those, so `.get("total", 1)` yielded 1. Preserve that.
                "total": int(row.get("total", 0)) or 1,
                "etym_examples": [] if pd.isna(row.get("etym_examples", "")) else str(row.get("etym_examples")).split("|"),
            }
            for _, row in caphi_flat.iterrows()
        }
    else:
        # fallback: build from the tidy table (without extras)
        caphi_coda_dict = {
            (row["Dialect"], row["CODA"], row["CAPHI"]): {
                "P(CODA|CAPHI)": row["P(CODA|CAPHI)"],
                "P(CAPHI|CODA)": row["P(CAPHI|CODA)"],
                "default mapping": False,
                "phon_count": 0, "etym_count": 0, "total": 1, "etym_examples": [],
            }
            for _, row in caphi_coda_probability_df.iterrows()
        }

    if os.path.exists(raw_flat_path):
        raw_flat = _read_tsv(raw_flat_path)
        raw_coda_dict = {
            (row["Dialect"], row["raw"], row["CODA"]): {
                "P(raw|CODA)": row.get("P(raw|CODA)", 0),
                "P(CODA|raw)": row.get("P(CODA|raw)", 0),
            }
            for _, row in raw_flat.iterrows()
        }
    else:
        raw_coda_dict = {
            (row["Dialect"], row["raw"], row["CODA"]): {
                "P(raw|CODA)": row["P(raw|CODA)"],
                "P(CODA|raw)": row["P(CODA|raw)"],
            }
            for _, row in raw_coda_probability_df.iterrows()
        }

    # 3) Dialect → raw → {CAPHI: prob}
    with open(os.path.join(output_dir, "P_CAPHI_given_ORTHO_by_dialect.json"), "r", encoding="utf-8") as f:
        dialect_dict = json.load(f)

    # 4) CAPHI letters universe (for CODA_SET = mappings.keys())
    caphi_table = _read_tsv(caphi_table_path)
    # `Letter` is CODA char; gather all unique letters present
    mappings = {letter: True for letter in caphi_table["Letter"].dropna().astype(str).unique()}

# --------- your original function, using loaded globals ----------
def compute_substitution_cost(letters, threshold: float = 0):
    """
    letters: [ (letter1, dialect1), (letter2, dialect2) ]
    Uses globals populated by load_distance_resources().
    Returns: (substitution_cost, best_path_dict)
    """
    # pull globals
    global caphi_coda_probability_df, raw_coda_probability_df
    global caphi_coda_dict, raw_coda_dict, dialect_dict, mappings

    if any(x is None for x in [caphi_coda_probability_df, raw_coda_probability_df, caphi_coda_dict, raw_coda_dict, dialect_dict, mappings]):
        raise RuntimeError("Distance resources not loaded. Call load_distance_resources() first.")

    dialect1 , dialect2 = [x[1] for x in letters]
    if dialect2 == 'CODA':
        dialect2 = dialect1
        dialect1 = 'CODA'
        letter1, letter2 = [x[0] for x in letters][::-1]
    else:
        letter1, letter2 = [x[0] for x in letters]

    # if the dialect is MSA (CODA baseline), compute without dialect constraint for letter1
    if ((dialect1 == 'CODA') & (letter1 != '-1') & (letter2 != '-1')):
        caphi_1 = list(set(caphi_coda_probability_df.loc[(caphi_coda_probability_df['CODA'] == letter1), 'CAPHI']))
    else:
        caphi_1 = list(set(caphi_coda_probability_df.loc[(caphi_coda_probability_df['CODA'] == letter1) & (caphi_coda_probability_df['Dialect'] == dialect1), 'CAPHI']))

    caphi_2 = list(set(caphi_coda_probability_df.loc[(caphi_coda_probability_df['CODA'] == letter2) & (caphi_coda_probability_df['Dialect'] == dialect2), 'CAPHI']))

    # if no CAPHI candidates (invalid character)
    if ((len(caphi_1) == 0) & (letter1 != '-1') & (letter2 != '-1')):
        return 1, {'coda': '-1', letter1: '-1'}
    if ((len(caphi_2) == 0) & (letter2 != '-1') & (letter1 != '-1')):
        return 1, {'coda': '-1', letter2: '-1'}
    if (letters[0] == letters[1]):
        return 0, {
            'coda': letters[0],
            letter2: max({caphi_coda_dict.get((dialect1, letters[0], a), {}).get("P(CODA|CAPHI)", 0)
                          * dialect_dict[dialect1].get(letters[0], {}).get(a, 0) for a in caphi_1})
        }

    # deletion/insert edge cases
    if ((letter1 == '-1') & (dialect1 == 'CODA')):
        prob_sum = 0
        best_path = None
        max_prob = 0
        for b in caphi_2:
            prob_etym = caphi_coda_dict.get((dialect2, letter2, b), {}).get("etym_count", 0) / (caphi_coda_dict.get((dialect2, letter2, b), {}).get("total", 1))
            if (prob_etym < 0.4):
                prob_1 = caphi_coda_dict.get((dialect2, '-1', b), {}).get("P(CODA|CAPHI)", 0) * dialect_dict.get(dialect2, {}).get(letter2, {}).get(b, 0)
                if prob_1 > max_prob:
                    max_prob = prob_1
                    best_path = {'coda': letter1, letter2: b}
                prob_sum += prob_1
        substitution_cost = 1 - prob_sum
        if (substitution_cost < threshold):
            substitution_cost = 0
        return (substitution_cost, best_path)

    elif ((letter2 == '-1') & (dialect1 != 'CODA')):
        prob_sum = 0
        best_path = None
        max_prob = 0
        for a in caphi_1:
            prob_etym = caphi_coda_dict.get((dialect1, letter1, a), {}).get("etym_count", 0) / (caphi_coda_dict.get((dialect1, letter1, a), {}).get("total", 1))
            if (prob_etym < 0.4):
                prob_1 = caphi_coda_dict.get((dialect1, '-1', a), {}).get("P(CODA|CAPHI)", 0) * dialect_dict.get(dialect1, {}).get(letter1, {}).get(a, 0)
                if prob_1 > max_prob:
                    max_prob = prob_1
                    best_path = {'coda': letter1, letter2: a}
                prob_sum += prob_1
        substitution_cost = 1 - prob_sum
        if (substitution_cost < threshold):
            substitution_cost = 0
        return (substitution_cost, best_path)

    elif ((letter1 == '-1') & (dialect1 != 'CODA')):
        prob_sum = 0
        best_path = None
        max_prob = 0
        for b in caphi_2:
            prob_etym = caphi_coda_dict.get((dialect2, letter2, b), {}).get("etym_count", 0) / (caphi_coda_dict.get((dialect2, letter2, b), {}).get("total", 1))
            if (prob_etym < 0.4):
                prob_1 = caphi_coda_dict.get((dialect2, '-1', b), {}).get("P(CODA|CAPHI)", 0) * dialect_dict.get(dialect2, {}).get(letter2, {}).get(b, 0)
                if prob_1 > max_prob:
                    max_prob = prob_1
                    best_path = {'coda': letter1, letter2: b}
                prob_sum += prob_1
        substitution_cost = 1 - prob_sum
        if (substitution_cost < threshold):
            substitution_cost = 0
        return (substitution_cost, best_path)

    elif ((letter2 == '-1') & (dialect1 == 'CODA')):
        substitution_cost = 1 - raw_coda_dict.get((dialect2, '-1', letter1), {}).get("P(raw|CODA)", 0)
        if (substitution_cost < threshold):
            substitution_cost = 0
        return (substitution_cost, {'coda': letter1, letter2: 'NA'})

    # CODA_CAPHI branch
    if (dialect1 == 'CODA'):
        caphi_2 = list(set(caphi_coda_probability_df.loc[(caphi_coda_probability_df['CODA'] == letter2), 'CAPHI']))
        prob_sum = 0
        best_prob = 0
        best_path = None
        for b in caphi_2:
            prob_etym = caphi_coda_dict.get((dialect2, letter2, b), {}).get("etym_count", 0) / (caphi_coda_dict.get((dialect2, letter2, b), {}).get("total", 1))
            if (prob_etym < 0.4):
                prob = caphi_coda_dict.get((dialect2, letter1, b), {}).get("P(CODA|CAPHI)", 0) * dialect_dict[dialect2].get(letter2, {}).get(b, 0)
                prob_sum += prob
                if prob > best_prob:
                    best_prob = prob
                    best_path = {'CODA': letter1, letter2: b}
        substitution_cost = 1 - prob_sum
        if (substitution_cost < threshold):
            substitution_cost = 0
        return substitution_cost, best_path

    # general case
    prob_sum = 0
    best_prob = 0
    best_path = None

    CODA_SET = mappings.keys()  # CODA letters universe from CAPHI table

    for coda in CODA_SET:
        for a in caphi_1:
            for b in caphi_2:
                prob_etym_1 = (caphi_coda_dict.get((dialect1, letter1, a), {}).get("etym_count", 0) /
                                (caphi_coda_dict.get((dialect1, letter1, a), {}).get("total", 1))
                                )
                if ((prob_etym_1 > 0.4) and (letter1 != coda)
                        and (not caphi_coda_dict.get((dialect1, letter1, a), {}).get("default mapping", False))):
                    prob_1 = 0
                elif ((prob_etym_1 > 0.4) and (letter1 == coda)):
                    prob_1 = dialect_dict[dialect1][letter1][a]
                else:
                    prob_1 = caphi_coda_dict.get((dialect1, coda, a), {}).get("P(CODA|CAPHI)", 0) \
                             * dialect_dict[dialect1].get(letter1, {}).get(a, 0)

                prob_etym_2 = (caphi_coda_dict.get((dialect2, letter2, b), {}).get("etym_count", 0) /
                                (caphi_coda_dict.get((dialect2, letter2, b), {}).get("total", 1)))
                if ((prob_etym_2 > 0.4) and (letter2 != coda)
                        and not (caphi_coda_dict.get((dialect2, letter2, b), {}).get("default mapping", False))):
                    prob_2 = 0
                elif ((prob_etym_2 > 0.4) and letter2 == coda):
                    prob_2 = dialect_dict[dialect2][letter2][b]
                else:
                    prob_2 = caphi_coda_dict.get((dialect2, coda, b), {}).get("P(CODA|CAPHI)", 0) \
                             * dialect_dict[dialect2].get(letter2, {}).get(b, 0)

                prod = prob_1 * prob_2
                prob_sum += prod
                if prod > best_prob:
                    best_prob = prod
                    best_path = {'coda': coda, letter1: a, letter2: b}

    substitution_cost = 1 - prob_sum
    if (substitution_cost < threshold):
        substitution_cost = 0
    return substitution_cost, best_path

# ---------- quick usage example ----------
if __name__ == "__main__":
    load_distance_resources()  # populates globals from ./output + CAPHI table
    letters = [("ظ", "SAL"), ("ض", "BEI")]
    cost, path = compute_substitution_cost(letters, threshold=0)
    print({"cost": cost, "path": path})
