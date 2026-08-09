# distance_function/precompute_probabilities.py
# Build the CODA<->CAPHI, ORTHO<->CODA and CAPHI|ORTHO probability tables (plus
# phonological/etymological counts) from MADAR-CODA CED alignments, the MADAR
# Lexicon and the CAPHI table. Writes TSV/JSON artifacts consumed by
# substitution_weight.py. Run from the repo root: python -m distance_function.precompute_probabilities

import pandas as pd
import numpy as np
import os

from utilities.preprocess_text import preprocess_text




# NOTE (see KNOWN_ISSUES.md): this list uses 'ALX'/'RIY'/'SFA' where the rest of the
# codebase and data/MADAR/MADAR.tsv use 'ALEX'/'ARI'/'SFX'. Kept verbatim from the
# thesis notebook: only the 6 core dialects (BEI/CAI/DOH/RAB/TUN + MSA) have CODA/CAPHI
# resources, so the 20 extra codes only ever key empty entries in dialect_dict below.
# Do NOT "fix" this in isolation — it would change the 26-dialect AGS numbers.
dialects_26 = ['MSA','BEI', 'ALX', 'AMM', 'ASW', 'ALE', 'CAI',
       'DAM', 'JER', 'SAL', 'DOH', 'RAB', 'TUN', 'ALG', 'BAG', 'BAS', 'BEN',
       'FES', 'JED', 'KHA', 'MOS', 'MUS', 'RIY', 'SAN', 'SFA', 'TRI']


# --- logging + output dir ---
import logging, json

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("precompute_probabilities")

OUTPUT_DIR = os.environ.get("AGS_OUTPUT_DIR", "output")  # reuse your existing ./output
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _save_df(df: pd.DataFrame, name: str):
    path = os.path.join(OUTPUT_DIR, name)
    df.to_csv(path, sep="\t", index=False)
    log.info("saved %s (%d rows)", path, len(df))
    return path

def _save_json(obj, name: str):
    path = os.path.join(OUTPUT_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    log.info("saved %s", path)
    return path

# ---------- path resolution helpers ----------

def _resolve_madar_corpus_base(madar_base_path: str) -> str:
    """
    Return a normalized directory that directly contains 'MADAR.corpus.MSA.tsv'.
    Tries a few sensible candidates based on the provided path.
    """
    candidates = [
        madar_base_path,  # as-given
        os.path.join(madar_base_path, "MADAR_Corpus"),  # if caller passed the parent
        os.path.join("data", "MADAR", "MADAR.Parallel-Corpora-Public-Version1.1-25MAR2021", "MADAR_Corpus"),  # repo default
    ]
    for p in candidates:
        msa = os.path.join(p, "MADAR.corpus.MSA.tsv")
        if os.path.exists(msa):
            return os.path.abspath(p)
    raise FileNotFoundError(
        "Could not locate 'MADAR.corpus.MSA.tsv'. Tried:\n  - " +
        "\n  - ".join(os.path.abspath(os.path.join(c, 'MADAR.corpus.MSA.tsv')) for c in candidates)
    )


def _resolve_madar_coda_base(madar_coda_base_path: str) -> str:
    """
    Return a normalized directory that directly contains 'train/' and 'test/' for MADAR-CODA.
    """
    candidates = [
        madar_coda_base_path,
        os.path.join("distance_function", "data", "MADAR-CODA"),  # repo default
    ]
    for p in candidates:
        if os.path.isdir(os.path.join(p, "train")) and os.path.isdir(os.path.join(p, "test")):
            return os.path.abspath(p)
    raise FileNotFoundError(
        "Could not locate MADAR-CODA 'train/' and 'test/' folders. Tried:\n  - " +
        "\n  - ".join(os.path.abspath(p) for p in candidates)
    )


# ---------- loaders ----------

def import_caphi_table_df(path: str = "distance_function/data/CAPHI/CAPHI-table.tsv") -> pd.DataFrame:
    caphi_table = pd.read_csv(path, sep="\t")
    caphi_table.drop(
        columns=["Unnamed: 0", "audio URL", "audio icon", "CAPHI sorting order", "CODA sorting order"],
        inplace=True,
    )
    return caphi_table


def import_madar_lexicon_df(path: str = "distance_function/data/MADAR_Lexicon_v1.0/MADAR_Lexicon_v1.0.tsv") -> pd.DataFrame:
    MADAR_lex = pd.read_csv(path, sep="\t")
    MADAR_lex.set_index("ID", inplace=True)
    return MADAR_lex


def import_madar_coda_df(
    madar_coda_base_path: str = "./data/MADAR-CODA",
    madar_base_path: str = "data/MADAR/MADAR.Parallel-Corpora-Public-Version1.1-25MAR2021/MADAR_Corpus",
) -> pd.DataFrame:
    # Resolve actual roots so we don't double-append pieces
    madar_corpus_root = _resolve_madar_corpus_base(madar_base_path)
    madar_coda_root = _resolve_madar_coda_base(madar_coda_base_path)

    MSA = pd.read_csv(os.path.join(madar_corpus_root, "MADAR.corpus.MSA.tsv"), delimiter="\t")

    BEIRUT = pd.read_csv(os.path.join(madar_coda_root, "train", "train.Beirut.tsv"), delimiter="\t")[
        ["sentID.BTEC", "raw", "CODA"]
    ].rename({"raw": "BEI_raw", "CODA": "BEI_CODA"}, axis=1)
    BEIRUT_test = pd.read_csv(os.path.join(madar_coda_root, "test", "test.Beirut.tsv"), delimiter="\t")[
        ["sentID.BTEC", "raw", "CODA"]
    ].rename({"raw": "BEI_raw", "CODA": "BEI_CODA"}, axis=1)
    BEIRUT = pd.concat([BEIRUT, BEIRUT_test])

    CAIRO = pd.read_csv(os.path.join(madar_coda_root, "train", "train.Cairo.tsv"), delimiter="\t")[
        ["sentID.BTEC", "raw", "CODA"]
    ].rename({"raw": "CAI_raw", "CODA": "CAI_CODA"}, axis=1)
    CAIRO_test = pd.read_csv(os.path.join(madar_coda_root, "test", "test.Cairo.tsv"), delimiter="\t")[
        ["sentID.BTEC", "raw", "CODA"]
    ].rename({"raw": "CAI_raw", "CODA": "CAI_CODA"}, axis=1)
    CAIRO = pd.concat([CAIRO, CAIRO_test])

    DOHA = pd.read_csv(os.path.join(madar_coda_root, "train", "train.Doha.tsv"), delimiter="\t")[
        ["sentID.BTEC", "raw", "CODA"]
    ].rename({"raw": "DOH_raw", "CODA": "DOH_CODA"}, axis=1)
    DOHA_test = pd.read_csv(os.path.join(madar_coda_root, "test", "test.Doha.tsv"), delimiter="\t")[
        ["sentID.BTEC", "raw", "CODA"]
    ].rename({"raw": "DOH_raw", "CODA": "DOH_CODA"}, axis=1)
    DOHA = pd.concat([DOHA, DOHA_test])

    RABAT = pd.read_csv(os.path.join(madar_coda_root, "train", "train.Rabat.tsv"), delimiter="\t")[
        ["sentID.BTEC", "raw", "CODA"]
    ].rename({"raw": "RAB_raw", "CODA": "RAB_CODA"}, axis=1)
    RABAT_test = pd.read_csv(os.path.join(madar_coda_root, "test", "test.Rabat.tsv"), delimiter="\t")[
        ["sentID.BTEC", "raw", "CODA"]
    ].rename({"raw": "RAB_raw", "CODA": "RAB_CODA"}, axis=1)
    RABAT = pd.concat([RABAT, RABAT_test])

    TUNIS = pd.read_csv(os.path.join(madar_coda_root, "train", "train.Tunis.tsv"), delimiter="\t")[
        ["sentID.BTEC", "raw", "CODA"]
    ].rename({"raw": "TUN_raw", "CODA": "TUN_CODA"}, axis=1)
    TUNIs_test = pd.read_csv(os.path.join(madar_coda_root, "test", "test.Tunis.tsv"), delimiter="\t")[
        ["sentID.BTEC", "raw", "CODA"]
    ].rename({"raw": "TUN_raw", "CODA": "TUN_CODA"}, axis=1)
    TUNIS = pd.concat([TUNIS, TUNIs_test])

    MADAR_CODA = pd.merge(BEIRUT, CAIRO, on="sentID.BTEC", how="outer")
    MADAR_CODA = pd.merge(MADAR_CODA, DOHA, on="sentID.BTEC", how="outer")
    MADAR_CODA = pd.merge(MADAR_CODA, RABAT, on="sentID.BTEC", how="outer")
    MADAR_CODA = pd.merge(MADAR_CODA, TUNIS, on="sentID.BTEC", how="outer")
    MADAR_CODA = pd.merge(MADAR_CODA, MSA, on="sentID.BTEC", how="inner")
    MADAR_CODA.set_index("sentID.BTEC", inplace=True)

    MADAR_CODA.loc[
        :,
        [
            "BEI_raw",
            "BEI_CODA",
            "CAI_raw",
            "CAI_CODA",
            "DOH_raw",
            "DOH_CODA",
            "RAB_raw",
            "RAB_CODA",
            "TUN_raw",
            "TUN_CODA",
        ],
    ] = MADAR_CODA.loc[
        :,
        [
            "BEI_raw",
            "BEI_CODA",
            "CAI_raw",
            "CAI_CODA",
            "DOH_raw",
            "DOH_CODA",
            "RAB_raw",
            "RAB_CODA",
            "TUN_raw",
            "TUN_CODA",
        ],
    ].map(preprocess_text)

    return MADAR_CODA


def make_raw_caphi_mappings():
    caphi_table = import_caphi_table_df()
    mappings = caphi_table[["Letter", "CAPHI"]].groupby("Letter").agg(list).to_dict()["CAPHI"]
    mappings["ي"].extend(mappings["َي"])
    mappings["ا"].extend("2")
    mappings["ا"].extend("e")
    raw_caphi_map = mappings.copy()
    raw_caphi_df = pd.DataFrame(columns=["Letter", "CAPHI"])
    return raw_caphi_map, raw_caphi_df


def import_alignment_files(file1: str, file2: str) -> pd.DataFrame:
    sentences = []
    with open(file1, "r", encoding="utf-8") as f1, open(file2, "r", encoding="utf-8") as f2:
        sentence_pairs = []
        for line1, line2 in zip(f1, f2):
            line1, line2 = line1.strip(), line2.strip()
            if line1 == "" and line2 == "":
                if sentence_pairs:
                    sentences.append(sentence_pairs)
                    sentence_pairs = []
            else:
                sentence_pairs.append((line1, line2))
        if sentence_pairs:
            sentences.append(sentence_pairs)
    return pd.DataFrame({"Aligned Sentences": sentences})


def add_alignments_to_MADAR_CODA_df(
    df: pd.DataFrame,
    alignment_files_base_path: str = "distance_function/data/MADAR-CODA/word_alignment",
) -> pd.DataFrame:
    df = df.copy()
    base = os.path.abspath(alignment_files_base_path)

    df["BEI_alignment"] = import_alignment_files(
        os.path.join(base, "BEI_raw.txt.align"),
        os.path.join(base, "BEI_CODA.txt.align"),
    )["Aligned Sentences"].tolist()

    df["CAI_alignment"] = import_alignment_files(
        os.path.join(base, "CAI_raw.txt.align"),
        os.path.join(base, "CAI_CODA.txt.align"),
    )["Aligned Sentences"].tolist()

    df["DOH_alignment"] = import_alignment_files(
        os.path.join(base, "DOH_raw.txt.align"),
        os.path.join(base, "DOH_CODA.txt.align"),
    )["Aligned Sentences"].tolist()

    df["RAB_alignment"] = import_alignment_files(
        os.path.join(base, "RAB_raw.txt.align"),
        os.path.join(base, "RAB_CODA.txt.align"),
    )["Aligned Sentences"].tolist()

    df["TUN_alignment"] = import_alignment_files(
        os.path.join(base, "TUN_raw.txt.align"),
        os.path.join(base, "TUN_CODA.txt.align"),
    )["Aligned Sentences"].tolist()

    return df


def load_madar_coda_with_alignments(
    madar_coda_base_path: str = "distance_function/data/MADAR-CODA",
    madar_base_path: str = "data/MADAR/MADAR.Parallel-Corpora-Public-Version1.1-25MAR2021/MADAR_Corpus",
    alignment_files_base_path: str = "distance_function/data/MADAR-CODA/word_alignment",
) -> pd.DataFrame:
    """
    Convenience wrapper: load MADAR CODA dataframe and attach alignment columns.
    Auto-resolves base paths to avoid duplicated segments.
    """
    df = import_madar_coda_df(
        madar_coda_base_path=madar_coda_base_path,
        madar_base_path=madar_base_path,
    )
    df = add_alignments_to_MADAR_CODA_df(df, alignment_files_base_path=alignment_files_base_path)
    return df

def arabic_caphi_alignment(source, target, mapping):
    target = target.split()
    m, n = len(source), len(target)
    dp = np.full((m + 1, n + 1), float('inf'))
    backtrack = np.empty((m + 1, n + 1), dtype=object)

    dp[0][0] = 0

    for j in range(1, n + 1):
        insertion_cost = 0.9 if target[j - 1] in {'e', 'o', 'i', 'a', 'u'} else 1
        dp[0][j] = dp[0][j - 1] + insertion_cost
        backtrack[0][j] = (0, j - 1, "insert")

    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + 1
        backtrack[i][0] = (i - 1, 0, "delete")

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            arabic_char = source[i - 1]
            caphi_char = target[j - 1]

            # Substitution
            cost = 1  # Default cost for invalid substitution
            if ((arabic_char in mapping) and (caphi_char in mapping.get(arabic_char, []))):
                cost = 0  # Valid mapping, no cost

            if dp[i - 1][j - 1] + cost < dp[i][j]:
                dp[i][j] = dp[i - 1][j - 1] + cost
                backtrack[i][j] = (i - 1, j - 1, "substitution")

            # Insertion
            insert_cost = 0.9 if caphi_char in {'e', 'o', 'i', 'a', 'u'} else 1

            if dp[i][j - 1] + insert_cost < dp[i][j]:
                dp[i][j] = dp[i][j - 1] + insert_cost
                backtrack[i][j] = (i, j - 1, "insert")

            #handling sunny lam ل
            if ((arabic_char == 'ل') and ('ال' in source) and ('l' not in target[:min(len(target)-1, 3) ])):
              del_cost = -0.001
            else:
              del_cost = 1
            # Deletion
            if dp[i - 1][j] + 1 < dp[i][j]:
                dp[i][j] = dp[i - 1][j] + del_cost
                backtrack[i][j] = (i - 1, j, "delete")

    # Backtrack to find the alignment
    i, j = m, n
    alignment = []
    while i > 0 or j > 0:
        if backtrack[i][j] is None:
            break
        prev_i, prev_j, operation = backtrack[i][j]
        if operation == "substitution":
            alignment.append((source[prev_i], target[prev_j]))
        elif operation == "insert":
            alignment.append((-1, target[prev_j]))
        elif operation == "delete":
            alignment.append((source[prev_i], -1))
        i, j = prev_i, prev_j

    #handling gemination as a postprocessing step
    alignment = alignment[::-1]

    prev_caphi = None
    for i, pair in enumerate(alignment):
      if ((pair[1] == prev_caphi) and (pair[0] == -1)):
        alignment[i] = alignment[i-1]
      elif((pair[1] == prev_caphi) and (alignment[i-1][0] == -1)):
        alignment[i-1] = alignment[i]
      prev_caphi = pair[1]

    return alignment

def raw_coda_char_alignment(source, target):
    m, n = len(source), len(target)
    dp = np.full((m + 1, n + 1), float('inf'))
    backtrack = np.empty((m + 1, n + 1), dtype=object)

    dp[0][0] = 0

    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + 1
        backtrack[0][j] = (0, j - 1, "insert")

    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + 1
        backtrack[i][0] = (i - 1, 0, "delete")

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            raw_char = source[i - 1]
            coda_char = target[j - 1]

            # Substitution
            cost = 0 if raw_char == coda_char else 1

            if dp[i - 1][j - 1] + cost < dp[i][j]:
                dp[i][j] = dp[i - 1][j - 1] + cost
                backtrack[i][j] = (i - 1, j - 1, "substitution")

            # Insertion
            if dp[i][j - 1] + 1 < dp[i][j]:
                dp[i][j] = dp[i][j - 1] + 1
                backtrack[i][j] = (i, j - 1, "insert")

            # Deletion
            if dp[i - 1][j] + 1 < dp[i][j]:
                dp[i][j] = dp[i - 1][j] + 1
                backtrack[i][j] = (i - 1, j, "delete")

    # Backtrack to find the alignment
    i, j = m, n
    alignment = []
    while i > 0 or j > 0:
        if backtrack[i][j] is None:
            break
        prev_i, prev_j, operation = backtrack[i][j]
        if operation == "substitution":
            alignment.append((source[prev_i], target[prev_j]))
        elif operation == "insert":
            alignment.append((-1, target[prev_j]))
        elif operation == "delete":
            alignment.append((source[prev_i], -1))
        i, j = prev_i, prev_j
    return alignment[::-1]


def calculate_alignment_probabilities(MADAR_lex, source_set = None, target_set = None):

    # Flatten the alignment tuples and extract Arabic letters and their CAPHI mappings
    all_alignments = []
    for _, row in MADAR_lex.dropna(subset=['Dialect', 'alignment']).iterrows():
        dialect = row['Dialect']
        for arabic_char, caphi_char in row['alignment']:
            # if source_set and target set are provided, check if alignments are in them
            if ((source_set is not None) and (arabic_char in source_set) and (target_set is not None) and (caphi_char in target_set)):
              all_alignments.append((dialect, arabic_char, caphi_char))
            elif ((source_set is not None) and (arabic_char in source_set)) and ((target_set is None)):
              all_alignments.append((dialect, arabic_char, caphi_char))
            elif ((target_set is not None) and (caphi_char in target_set)) and ((source_set is None)):
              all_alignments.append((dialect, arabic_char, caphi_char))
            elif ((source_set is None) and (target_set is None)):
              all_alignments.append((dialect, arabic_char, caphi_char))

    # Convert to DataFrame
    alignment_df = pd.DataFrame(all_alignments, columns=['Dialect', 'CODA', 'CAPHI'])

    # Compute counts
    alignment_counts = alignment_df.groupby(['Dialect', 'CODA', 'CAPHI']).size().reset_index(name='CODA-CAPHI count')
    CODA_counts = alignment_df.groupby(['Dialect', 'CODA']).size().reset_index(name='CODA count')
    CAPHI_counts = alignment_df.groupby(['Dialect', 'CAPHI']).size().reset_index(name='CAPHI count')
    # Merge to calculate probabilities
    alignment_counts = alignment_counts.merge(CODA_counts, on=['Dialect', 'CODA'])
    alignment_counts = alignment_counts.merge(CAPHI_counts, on=['Dialect', 'CAPHI'])

    prob_df = alignment_counts.copy()
    prob_df['P(CODA|CAPHI)'] = prob_df['CODA-CAPHI count'] / prob_df['CAPHI count']
    prob_df['P(CAPHI|CODA)'] = prob_df['CODA-CAPHI count'] / prob_df['CODA count']

    # Drop unnecessary columns
    prob_df = prob_df[['Dialect', 'CODA', 'CAPHI', 'P(CODA|CAPHI)', 'P(CAPHI|CODA)']]
    prob_df.loc[:, ['P(CODA|CAPHI)', 'P(CAPHI|CODA)']] = prob_df.loc[:, ['P(CODA|CAPHI)', 'P(CAPHI|CODA)']].map(lambda x: round(x, 4))
    return prob_df




def prob_caphi_given_ortho(raw, caphi, dialect):
  #sum over all codas or coda given raw and caphi given coda
  prob_sum = 0
  for coda in raw_coda_probability_df['CODA'].unique():
    prob_sum += raw_coda_dict.get((dialect, raw, coda), {}).get("P(CODA|raw)", 0) * caphi_coda_dict.get((dialect, coda, caphi), {}).get("P(CAPHI|CODA)", 0)
  return prob_sum

def prob_ortho_given_caphi(raw, caphi, dialect):
  #sum over all codas or coda given raw and caphi
  prob_sum = 0
  for coda in raw_coda_probability_df['CODA'].unique():
    prob_sum += raw_coda_dict.get((dialect, raw, coda), {}).get("P(raw|CODA)", 0) * caphi_coda_dict.get((dialect, coda, caphi), {}).get("P(CODA|CAPHI)", 0)
  return prob_sum




def aggregate_alignments_by_dialect(df):
    df = df.dropna(subset=['alignment'])
    coda_word = df['CODA'].iloc[0]
    alignment_list = [(char, {}) for char in coda_word]
    for _, row in df.iterrows():
        dialect = row["Dialect"]
        curr_idx = 0
        #filter_out gemination
        alignment = row['alignment'].copy()
        current_coda_word_idx = 0
        while (current_coda_word_idx < len(row['CODA'])):
          if (alignment[current_coda_word_idx][0] == row['CODA'][current_coda_word_idx]):
            current_coda_word_idx += 1
            continue
          else:
            alignment.pop(current_coda_word_idx)
            current_coda_word_idx += 1
        alignment = alignment[: current_coda_word_idx]
        for coda_char, caphi_char in alignment:
            if(coda_char == -1):
              continue
        for (coda_char, caphi_char) in alignment:
            alignment_list[curr_idx][1][dialect] = caphi_char
            curr_idx += 1

    return alignment_list




if __name__ == '__main__':
    MADAR_CODA = load_madar_coda_with_alignments()

    #################CAPHI-CODA Probability###########################
    raw_CODA_BEI = [x for y in MADAR_CODA['BEI_alignment'] for x in y]
    raw_CODA_CAI = [x for y in MADAR_CODA['CAI_alignment'] for x in y]
    raw_CODA_DOH = [x for y in MADAR_CODA['DOH_alignment'] for x in y]
    raw_CODA_RAB = [x for y in MADAR_CODA['RAB_alignment'] for x in y]
    raw_CODA_TUN = [x for y in MADAR_CODA['TUN_alignment'] for x in y]
    raw_CODA_df = pd.DataFrame({'raw': [x[0] for x in raw_CODA_BEI], 'CODA': [x[1] for x in raw_CODA_BEI], 'Dialect': 'BEI'})
    raw_CODA_df = pd.concat([raw_CODA_df, pd.DataFrame({'raw': [x[0] for x in raw_CODA_CAI], 'CODA': [x[1] for x in raw_CODA_CAI], 'Dialect': 'CAI'})])
    raw_CODA_df = pd.concat([raw_CODA_df, pd.DataFrame({'raw': [x[0] for x in raw_CODA_DOH], 'CODA': [x[1] for x in raw_CODA_DOH], 'Dialect': 'DOH'})])
    raw_CODA_df = pd.concat([raw_CODA_df, pd.DataFrame({'raw': [x[0] for x in raw_CODA_RAB], 'CODA': [x[1] for x in raw_CODA_RAB], 'Dialect': 'RAB'})])
    raw_CODA_df = pd.concat([raw_CODA_df, pd.DataFrame({'raw': [x[0] for x in raw_CODA_TUN], 'CODA': [x[1] for x in raw_CODA_TUN], 'Dialect': 'TUN'})])
    ################

    MADAR_lex = import_madar_lexicon_df()
    raw_CODA_CAPHI = pd.merge(raw_CODA_df, MADAR_lex, on=['CODA', 'Dialect'], how='inner')
    raw_CODA_CAPHI[raw_CODA_CAPHI.apply(lambda x: x['raw'] != x['CODA'], axis=1)]
    raw_CODA_CAPHI[['raw', 'CODA', 'CAPHI', 'Dialect', 'Example', 'English']]
    # raw_CODA_CAPHI.to_csv('/content/drive/MyDrive/Thesis/raw_CODA_CAPHI.tsv', sep='\t', index=False)

    ################
    caphi_table = import_caphi_table_df()
    mappings = caphi_table[['Letter', 'CAPHI']].groupby('Letter').agg(list).to_dict()['CAPHI']
    mappings['ي'].extend(mappings['َي'])
    mappings['ا'].extend('2')
    mappings['ا'].extend('e')
    MADAR_lex['alignment'] = MADAR_lex.apply(lambda row: arabic_caphi_alignment(row['CODA'], row['CAPHI'], mappings) if not pd.isna(row['CAPHI']) and not pd.isna(row['CODA']) else np.nan, axis=1)
    # MADAR_lex.to_csv('/content/drive/MyDrive/Thesis/Data/MADAR_Lexicon_v1.0/MADAR_Lexicon_v1.0_aligned.tsv', sep='\t', index=False)
    ##################


    CAPHI_CODA_probability_df = calculate_alignment_probabilities(MADAR_lex, source_set= (list(caphi_table['Letter'].unique()) + ['-1', -1]), target_set=(list(caphi_table['CAPHI'].unique())+ ['-1' , -1]))
    CAPHI_CODA_probability_df.columns = ['Dialect', 'CODA', 'CAPHI', 'P(CODA|CAPHI)', 'P(CAPHI|CODA)']
    CAPHI_CODA_probability_df['default mapping'] = CAPHI_CODA_probability_df.apply(lambda row: 'default' in caphi_table.loc[(caphi_table['Letter'] == row['CODA']) & (caphi_table['CAPHI'] == row['CAPHI'])]['Mapping'].tolist(), axis=1)
    # faster lookup
    caphi_coda_dict = {
        (row["Dialect"], row["CODA"], row["CAPHI"]): {"P(CODA|CAPHI)": row["P(CODA|CAPHI)"], "P(CAPHI|CODA)": row["P(CAPHI|CODA)"], 'default mapping': row['default mapping']}
        for _, row in CAPHI_CODA_probability_df.iterrows()
    }


    ################# ORTHO|CODA Probability###########################
    raw_CODA_df['alignment'] = raw_CODA_df.apply(lambda row: raw_coda_char_alignment(row['raw'], row['CODA']), axis=1)
    raw_CODA_df.map(str)
    raw_coda_probability_df = calculate_alignment_probabilities(raw_CODA_df, source_set= (list(caphi_table['Letter'].unique()) + ['-1', -1]), target_set=(list(caphi_table['Letter'].unique())+ ['-1', -1]))
    raw_coda_probability_df.columns = ['Dialect', 'raw', 'CODA', 'P(raw|CODA)', 'P(CODA|raw)']
    raw_coda_dict = {
    (row['Dialect'], row["raw"], row["CODA"]): {"P(raw|CODA)": row["P(raw|CODA)"],"P(CODA|raw)": row["P(CODA|raw)"]}
    for _, row in raw_coda_probability_df.iterrows()
    }


    ################### P(CAPHI|ORTHO) #######################
    dialects_25 = dialects_26.copy()
    dialects_25.remove('MSA')

    # BEI_raw_caphi_mappings = dict()
    # CAI_raw_caphi_mappings = dict()
    # DOH_raw_caphi_mappings = dict()
    # RAB_raw_caphi_mappings = dict()
    # TUN_raw_caphi_mappings = dict()
    # dialect_dict = {'BEI': BEI_raw_caphi_mappings, 'CAI': CAI_raw_caphi_mappings, 'DOH': DOH_raw_caphi_mappings, 'RAB': RAB_raw_caphi_mappings, 'TUN': TUN_raw_caphi_mappings}
    dialect_dict = {dialect : {} for dialect in dialects_25}
    dialects_5 = ['BEI', 'CAI', 'DOH', 'RAB', 'TUN']
    #if its one of dialects_5 we compute p(CAPHI|ORTHO) using MADAR-CODA, otherwise, if it's from the other dialects we just use P(CAPHI|CODA)
    for dialect in dialects_25:
        if dialect in dialects_5:
            for raw in mappings:
                caphi_set = list(set(CAPHI_CODA_probability_df.loc[(CAPHI_CODA_probability_df['Dialect'] == dialect) & (CAPHI_CODA_probability_df['CODA'] == raw), 'CAPHI']))
                for caphi in caphi_set:
                    total_prob = sum([prob_caphi_given_ortho(raw, caphi, dialect) for caphi in caphi_set])
                    dialect_dict[dialect][raw] = {caphi: prob_caphi_given_ortho(raw, caphi, dialect)/total_prob if total_prob else 1/len(caphi_set) for caphi in caphi_set}
        else :
            for raw in mappings:
                caphi_set = list(set(CAPHI_CODA_probability_df.loc[(CAPHI_CODA_probability_df['Dialect'] == dialect) & (CAPHI_CODA_probability_df['CODA'] == raw), 'CAPHI']))
                for caphi in caphi_set:
                    dialect_dict[dialect][raw] = {caphi: caphi_coda_dict.get((dialect, raw, caphi), {}).get("P(CAPHI|CODA)", 0) for caphi in caphi_set}

    ############### Prob PHON, ETYM ################################
    phon_etym_df = MADAR_lex[['Concept_ID', 'Dialect', 'CODA', 'CAPHI', 'alignment']]
    codas = phon_etym_df['CODA'].unique()      
    alignments_dict = dict()
    for coda in codas:
        try:
            df = phon_etym_df[phon_etym_df['CODA'] == coda]
            formatted_aggregated_alignments = aggregate_alignments_by_dialect(df)
            alignments_dict[coda] = formatted_aggregated_alignments
        except:
            print(coda)
    coda_to_dafault_caphi_dict = dict()
    for coda in caphi_table['Letter'].unique():
        default_caphi_set = set(caphi_table.loc[(caphi_table['Letter'] == coda) & (caphi_table['Mapping'] == 'default'), 'CAPHI'])
        coda_to_dafault_caphi_dict[coda] = default_caphi_set


    for coda, alignment in alignments_dict.items():
        for coda_char, caphi_align_dict in alignment:
            #if a coda char is mapped to different default and non_default caphi chars, then all mappings are etymological, and the default ones are phonological too
            #else all mappings are phonological
            caphi_set = set(caphi_align_dict.values())
            if caphi_set.issubset(coda_to_dafault_caphi_dict.get(coda_char, {})):
                for dialect, caphi_char in caphi_align_dict.items():
                    if (dialect, coda_char, caphi_char) in caphi_coda_dict:
                        caphi_coda_dict[(dialect, coda_char, caphi_char)]['total'] = caphi_coda_dict[(dialect, coda_char, caphi_char)].get('total', 0) + 1
                        caphi_coda_dict[(dialect, coda_char, caphi_char)]['phon_count'] = caphi_coda_dict[(dialect, coda_char, caphi_char)].get('phon_count', 0) + 1
            else:
                for dialect, caphi_char in caphi_align_dict.items():
                    updated = False
                    if (dialect, coda_char, caphi_char) in caphi_coda_dict:
                        if(len(caphi_set)> 1):
                            updated = True
                            caphi_coda_dict[(dialect, coda_char, caphi_char)]['total'] = caphi_coda_dict[(dialect, coda_char, caphi_char)].get('total', 0) + 1
                            caphi_coda_dict[(dialect, coda_char, caphi_char)]['etym_count'] = caphi_coda_dict[(dialect, coda_char, caphi_char)].get('etym_count', 0) + 1
                            caphi_coda_dict[(dialect, coda_char, caphi_char)]['etym_examples'] = caphi_coda_dict[(dialect, coda_char, caphi_char)].get('etym_examples', []) + [coda]
                    if caphi_char in coda_to_dafault_caphi_dict.get(coda_char, {}):
                        updated = True
                        caphi_coda_dict[(dialect, coda_char, caphi_char)]['phon_count'] = caphi_coda_dict[(dialect, coda_char, caphi_char)].get('phon_count', 0) + 1
                    if updated:
                        caphi_coda_dict[(dialect, coda_char, caphi_char)]['total'] = caphi_coda_dict[(dialect, coda_char, caphi_char)].get('total', 0) + 1

    # --- persist everything the distance function will need ---

    # 1) Debug/inspection dumps (optional but handy)
    _save_df(MADAR_lex.reset_index(drop=False), "MADAR_Lexicon_with_alignments.tsv")
    _save_df(raw_CODA_df, "RAW_CODA_char_alignments.tsv")

    # 2) Probability tables (primary inputs downstream)
    caphi_coda_path = _save_df(CAPHI_CODA_probability_df, "CAPHI_CODA_probabilities.tsv")
    raw_coda_path   = _save_df(raw_coda_probability_df,   "RAW_CODA_probabilities.tsv")

    # 3) Dicts → flat tables (so distance function can read without Python pickles)

    # 3a) caphi_coda_dict (Dialect, CODA, CAPHI → probs + flags)
    caphi_rows = []
    for (d, coda, caphi), vals in caphi_coda_dict.items():
        caphi_rows.append({
            "Dialect": d,
            "CODA": coda,
            "CAPHI": caphi,
            "P(CODA|CAPHI)": vals.get("P(CODA|CAPHI)", 0),
            "P(CAPHI|CODA)": vals.get("P(CAPHI|CODA)", 0),
            "default_mapping": vals.get("default mapping", False),
            "phon_count": vals.get("phon_count", 0),
            "etym_count": vals.get("etym_count", 0),
            "total": vals.get("total", 0),
            "etym_examples": "|".join(vals.get("etym_examples", [])) if vals.get("etym_examples") else ""
        })
    caphi_coda_flat = pd.DataFrame(caphi_rows)
    _save_df(caphi_coda_flat, "CAPHI_CODA_probabilities_flat.tsv")

    # 3b) raw_coda_dict (Dialect, raw, CODA → probs)
    raw_rows = []
    for (d, raw_char, coda_char), vals in raw_coda_dict.items():
        raw_rows.append({
            "Dialect": d,
            "raw": raw_char,
            "CODA": coda_char,
            "P(raw|CODA)": vals.get("P(raw|CODA)", 0),
            "P(CODA|raw)": vals.get("P(CODA|raw)", 0),
        })
    raw_coda_flat = pd.DataFrame(raw_rows)
    _save_df(raw_coda_flat, "RAW_CODA_probabilities_flat.tsv")

    # 4) dialect_dict (Dialect → raw → {caphi: prob})
    #    Keep JSON (lossless) and also a flat TSV for convenience.
    _save_json(dialect_dict, "P_CAPHI_given_ORTHO_by_dialect.json")

    dialect_rows = []
    for d, raw_map in dialect_dict.items():
        for raw_char, caphi_map in raw_map.items():
            for caphi_char, prob in (caphi_map or {}).items():
                dialect_rows.append({
                    "Dialect": d,
                    "raw": raw_char,
                    "CAPHI": caphi_char,
                    "P(CAPHI|ORTHO)": prob,
                })
    if dialect_rows:
        _save_df(pd.DataFrame(dialect_rows), "P_CAPHI_given_ORTHO_by_dialect.tsv")

    log.info("all artifacts saved to: %s", os.path.abspath(OUTPUT_DIR))
