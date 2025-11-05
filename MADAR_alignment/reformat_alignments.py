
import os
import sys
import argparse
import logging
import pandas as pd
from copy import deepcopy
from collections import Counter

# NEW: tqdm for progress bars
from tqdm import tqdm
tqdm.pandas()  # enables .progress_apply / .progress_map

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utilities.preprocess_text import preprocess_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("run_alignment")

dialects_26 = [
    "BEI", "ALEX", "AMM", "ASW", "ALE", "CAI", "DAM", "JER", "SAL",
    "MSA", "DOH", "RAB", "TUN", "ALG", "BAG", "BAS", "BEN", "FES",
    "JED", "KHA", "MOS", "MUS", "ARI", "SAN", "SFX", "TRI"
]

def import_MADAR():
    """
    Load MADAR TSV and preprocess only the dialect columns with your `preprocess_text`.
    Keeps NaNs as-is. Fails early if expected columns are missing.
    """
    path = "data/MADAR/MADAR.tsv"
    logger.info(f"Loading MADAR from: {path}")
    MADAR_df = pd.read_csv(path, sep="\t", encoding="utf-8")

    # Sanity-check expected columns
    missing = [c for c in dialects_26 if c not in MADAR_df.columns]
    if missing:
        raise ValueError(
            f"Missing MADAR columns: {missing}. "
            f"Present sample: {sorted(MADAR_df.columns.tolist())[:12]}..."
        )

    logger.info("Preprocessing dialect columns with `preprocess_text` (NaNs preserved)")
    # PROGRESS: column-wise with progress bars
    for col in tqdm(dialects_26, desc="Preprocessing columns"):
        MADAR_df[col] = MADAR_df[col].progress_apply(
            lambda x: preprocess_text(x) if pd.notnull(x) else x
        )

    logger.info(f"MADAR loaded. Rows: {len(MADAR_df)}")
    return MADAR_df


def parse_args():
    p = argparse.ArgumentParser(description="Reformat alignments and merge many-to-one and one-to-many mappings")
    p.add_argument("--alignment_indices_file", required=False, default="output/finetuned_awesome_align_output_MADAR_26_idx.txt", help="Path to awesome_align output indices")
    p.add_argument("--output_file", required=False, default='data/MADAR_sent_alignments.tsv' ,help="path to reformatted output")
    return p.parse_args()


def pivot_alignments_around_MSA(df):
    alignment_indices_path = 'output/finetuned_awesome_align_output_MADAR_26_idx.txt'
    index_dialect_path = 'output/id_dialect.txt'
    MADAR_df = df

    alignment_indices = []
    with open(alignment_indices_path, 'r', encoding='utf-8') as alignment_file:
        alignment_indices = alignment_file.readlines()
        # PROGRESS: parsing lines
        alignment_indices = [x.removesuffix('\n') for x in tqdm(alignment_indices, desc="Reading alignment lines")]
        alignment_indices = [x.split() for x in tqdm(alignment_indices, desc="Splitting alignments")]
        alignment_indices = [
            [tuple([int(l) for l in x.split('-')]) for x in z]
            for z in tqdm(alignment_indices, desc="Converting to tuples")
        ]

    index_dialect = []
    with open(index_dialect_path, 'r', encoding= 'utf-8') as index_dialect_file:
        index_dialect = index_dialect_file.readlines()
        # PROGRESS: parsing id→dialect
        index_dialect = [tuple(x.removesuffix('\n').split('\t')) for x in tqdm(index_dialect, desc="Reading id_dialect")]
        index_dialect = [(int(x), y) for x,y in tqdm(index_dialect, desc="Casting ids")]

    alignment_indices_dic = {i : {} for i in MADAR_df.index}
    # PROGRESS: building dictionary
    for i in tqdm(range(len(alignment_indices)), desc="Building alignment dict"):
        alignment_indices_dic[index_dialect[i][0]][index_dialect[i][1]] = alignment_indices[i]

    MADAR_df['MSA_alignment_dictionary'] = alignment_indices_dic.values()
    return MADAR_df


# handle one to many and reformat to {MSA word: {Dialect: Alignments for Dialect in Dialects}}
def reformat_alignment_dictionary(alignment_dic):
    reformatted = {k : {} for k in alignment_dic}
    for target, pairs in alignment_dic.items():
        for pair in pairs:
            source_idx, target_idx = pair
            if source_idx not in reformatted[target]:
                reformatted[target][(source_idx,)] = set()
            reformatted[target][(source_idx,)].add(target_idx)
            sorted(reformatted[target][(source_idx,)])
    return reformatted


# map reformatted index dictionary to words
def map_indices_to_words(x, source):
    word_dic = deepcopy(x[source+'_idx_mapping'])
    for target, mapping in word_dic.items():
        for source_idx, target_idxs in mapping.items():
            target_words = [x[target].split()[idx] for idx in target_idxs]
            word_dic[target][source_idx] = target_words
        keys = list(word_dic[target].keys())
        for key in keys:
            word_dic[target][" ".join([x[source].split()[i] for i in key])] = word_dic[target].pop(key)
    return word_dic


# This is different alignment logic that better handles many-MSA-to-one-DA mappings
# only requires MSA alignment (aligns around MSA axis by handling one-to-many first then many-to-one)

def create_alignment_table(row, return_word_df= False, dialects_ = ["MSA","BEI", "CAI", "TUN", "DOH", "RAB"]):
    """
    Step 1: Create a table where each MSA word is a column,
    and the corresponding dialect words are placed in the respective rows.
    """
    dialects = dialects_.copy()
    if 'MSA' in dialects:
        dialects.remove('MSA')
    # Extract the alignment dictionary from the row
    alignment_dict = row["MSA_alignment_dictionary"]

    # Initialize table with MSA words as columns
    alignment_table = {i: {dialect : [] for dialect in dialects} for i in range(len(row['MSA'].split()))}

    # Populate dialect alignments
    for dialect in dialects:
        mappings = alignment_dict.get(dialect, {})
        for msa_idx, dialect_idx in mappings:
            alignment_table[msa_idx].setdefault(dialect, []).append(dialect_idx)
    alignment_df = pd.DataFrame.from_dict(alignment_table, orient="index")
    alignment_df = alignment_df.sort_index()
    if return_word_df:
        word_df = pd.DataFrame(columns=dialects + ['MSA'])
        for k, v in alignment_table.items():
            word_dic = {dialect: ' '.join( [row[dialect].split()[id] for id in sorted(v[dialect])]) for dialect in v }
            word_dic.update({'MSA': row['MSA'].split()[k]})
            word_df.loc[k] = word_dic
            print(alignment_df)
        return alignment_df, word_df
    return alignment_df


def merge_many_to_one(alignment_df, madar_row = None, return_word_df = False, dialects_ = ["MSA","BEI", "CAI", "TUN", "DOH", "RAB"]):
      """
      Step 2: Resolve many-to-one mappings by merging consecutive MSA words
      that align to the same phrase in at least one dialect.
      """
      dialects = dialects_.copy()
      if 'MSA' in dialects:
          dialects.remove('MSA')
      merged_rows = []
      current_segment = None

      alignment_df= alignment_df.map(lambda x: [] if (isinstance(x, float) and pd.isna(x)) else x)

      for i, row in alignment_df.iterrows():
          if current_segment is None:
              current_segment = row.to_dict()
              current_segment["MSA"] = [i, i]
          else:
              # Check if at least one dialectal column matches the previous row
              same_alignment = False
              for col in dialects:
                  if (row[col]!= []) and (set(row[col]) == set(current_segment[col])):
                      same_alignment = True
                      break
              if same_alignment:
                  current_segment["MSA"][1] = i  # Extend the segment
                  for col in dialects:
                     current_segment[col] = list(set(current_segment[col] + row[col]))
              else:
                  merged_rows.append(current_segment)
                  current_segment = row.to_dict()
                  current_segment["MSA"] = [i, i]

      if current_segment:
          merged_rows.append(current_segment)

      merged_df = pd.DataFrame(merged_rows)
      merged_df["MSA"] = merged_df["MSA"].apply(lambda x: list(range(x[0], x[1] + 1)))

      if return_word_df:
        if madar_row is None:
          raise ValueError("madar row must be provided if return_word_df is True")
        word_df = pd.DataFrame(columns=dialects + ['MSA'])
        for i, row in merged_df.iterrows():
          word_dic = {dialect: ' '.join([madar_row[dialect].split()[id] for id in sorted(row[dialect])]) if (row[dialect] != []) else None for dialect in dialects}
          word_dic.update({'MSA': ' '.join([madar_row['MSA'].split()[id] for id in sorted(row['MSA'])])})
          word_df.loc[i] = word_dic
        return merged_df, word_df
      return merged_df


def extract_phrases(word_df, dialects_ = ["MSA","BEI", "CAI", "TUN", "DOH", "RAB"]):
    """
    Step 3: Extract all phrases from the final table into a structured dataframe.
    """
    dialects = dialects_.copy()
    extracted_rows = []
    for _, row in word_df.iterrows():
        for dialect in dialects:
            if isinstance(row[dialect], str) and row[dialect]:
                phrase = row[dialect]
                new_row = {'word': phrase, 'dialect': dialect}
                new_row.update({col: row[col] for col in dialects})
                extracted_rows.append(new_row)

    return pd.DataFrame(extracted_rows)


def extract_word_alignments_pipeline(df, dialects_ = [ 'MSA', 'BEI', 'CAI', 'TUN', 'DOH', 'RAB']):
    dialects = dialects_.copy()
    columns = ['word', 'dialect'] + dialects

    # Collect chunks, then concat ONCE at the end (avoid O(n^2))
    chunks = []

    # progress over rows
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting per-row alignments", smoothing=0):
        alignment_df = create_alignment_table(row, dialects_=dialects)
        _, aligned_word_df = merge_many_to_one(alignment_df, row, return_word_df=True, dialects_=dialects)
        extracted_alignment_df = extract_phrases(aligned_word_df, dialects_=dialects)
        chunks.append(extracted_alignment_df)

    if not chunks:
        return pd.DataFrame(columns=columns)

    return pd.concat(chunks, ignore_index=True, copy=False)


def group_alignment_df(df, dialects_ = ['MSA', 'BEI', 'CAI', 'TUN', 'DOH', 'RAB']):
    # group by list and count of each
    dialects = dialects_.copy()

    # PROGRESS: normalization step
    df['normalized_word'] = df['word'].progress_map(lambda x: preprocess_text(x, True))

    # Return plain dicts (string->count), ignoring None/empty/non-strings
    def _count_terms(series: pd.Series) -> dict:
        vals = [v for v in series if isinstance(v, str) and v]
        return dict(Counter(vals))

    aggregations = {dialect: _count_terms for dialect in dialects}
    grouped = df.groupby(['normalized_word', 'dialect']).agg(aggregations).reset_index()
    return grouped


if __name__ == "__main__":
    MADAR_df = import_MADAR()
    MADAR_df = pivot_alignments_around_MSA(MADAR_df)

    # PROGRESS: dict reformat
    MADAR_df.loc[:, ['MSA_idx_mapping']] = MADAR_df['MSA_alignment_dictionary'].progress_map(reformat_alignment_dictionary)

    # PROGRESS: row-wise mapping
    MADAR_df.loc[:,['MSA_word_mapping']] = MADAR_df.progress_apply(lambda x: map_indices_to_words(x, 'MSA'), axis=1)

    all_word_df = extract_word_alignments_pipeline(MADAR_df, dialects_= dialects_26)
    grouped_alignment_df = group_alignment_df(all_word_df, dialects_= dialects_26)
    #
    grouped_alignment_df.to_csv('output/MADAR_reformatted_word_alignments.tsv', sep='\t')
