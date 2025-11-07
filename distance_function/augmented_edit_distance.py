#!/usr/bin/env python3
# distance_function/compute_distance.py
# Reordered & fixed: avoids SettingWithCopyWarning and KeyError('index').

from Levenshtein import distance as levenshtein_distance  # editops unused
import sys
import os
import pandas as pd
import ast
import numpy as np

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utilities.preprocess_text import preprocess_text
from substitution_weight import (
    load_distance_resources,  # populates globals used by compute_substitution_cost
    compute_substitution_cost,
)

# ---------------- constants ----------------
dialects_26 = [
    "MSA", "BEI", "ALEX", "AMM", "ASW", "ALE", "CAI",
    "DAM", "JER", "SAL", "DOH", "RAB", "TUN", "ALG",
    "BAG", "BAS", "BEN", "FES", "JED", "KHA", "MOS",
    "MUS", "ARI", "SAN", "SFX", "TRI"
]

# ------------- helpers -----------------
def expand_alignments_with_distance(alignment_df: pd.DataFrame, source="MSA", target="DOH") -> pd.DataFrame:
    """
    Expand MSA and dialect alignments into rows and compute Levenshtein edit distances.

    alignment_df columns contain 'dialect' and for each dialect, a dict of {word: freq}, including 'MSA'.
    We pick the MSA word with highest frequency and pair against all target-dialect words.
    """
    expanded_data = []
    df = alignment_df[alignment_df["dialect"] == source]
    for _, row in df.iterrows():
        # pick highest-frequency MSA word
        if isinstance(row["MSA"], dict) and row["MSA"]:
            msa_word = max(row["MSA"], key=row["MSA"].get)
        else:
            continue

        dialect_mappings = row.get(target)
        if isinstance(dialect_mappings, dict):
            for dialect_word, freq in dialect_mappings.items():
                if dialect_word:
                    edit_dist = levenshtein_distance(
                        preprocess_text(msa_word, True),
                        preprocess_text(dialect_word, True),
                    )
                    max_len = max(len(msa_word.replace(" ", "")), len(dialect_word.replace(" ", "")))
                    norm = edit_dist / max_len if max_len > 0 else 0.0
                    expanded_data.append([msa_word, dialect_word, target, edit_dist, norm, freq])

    return pd.DataFrame(
        expanded_data,
        columns=["MSA", target, "Dialect", "Edit_Distance", "Normalized_Edit_Distance", "Frequency"],
    )

def extract_prefix(row: pd.Series, dialect: str):
    msa_word = row["MSA"]
    dialect_word = row[dialect]

    normalized_msa_word = preprocess_text(msa_word, True)
    normalized_dialect_word = preprocess_text(dialect_word, True)

    if normalized_dialect_word.endswith(normalized_msa_word) and len(dialect_word) - len(msa_word) in [1, 2, 3]:
        return (("prefix", dialect, "MSA"), dialect_word[:-len(msa_word)])
    elif normalized_msa_word.endswith(normalized_dialect_word) and len(msa_word) - len(dialect_word) in [1, 2, 3]:
        return (("prefix", "MSA", dialect), msa_word[:-len(dialect_word)])
    return None

def extract_suffix(row: pd.Series, dialect: str):
    msa_word = row["MSA"]
    dialect_word = row[dialect]

    normalized_msa_word = preprocess_text(msa_word, True)
    normalized_dialect_word = preprocess_text(dialect_word, True)

    if normalized_dialect_word.startswith(normalized_msa_word) and len(dialect_word) - len(msa_word) in [1, 2, 3]:
        return (("suffix", dialect, "MSA"), dialect_word[len(msa_word):])
    elif normalized_msa_word.startswith(normalized_dialect_word) and len(msa_word) - len(dialect_word) in [1, 2, 3]:
        return (("suffix", "MSA", dialect), msa_word[len(dialect_word):])
    return None

# ------------- main class -----------------
class DistanceCalculator:
    def __init__(self, compute_substitution_cost_func, scaler="exponential", threshold=0,
                 prefixes=None, suffixes=None):
        self.compute_substitution_cost = compute_substitution_cost_func
        self.threshold = threshold
        self.scaler = scaler
        self.prefixes_dict = prefixes or {}
        self.suffixes_dict = suffixes or {}
        self.cost_cache: dict = {}
        self.distance_cache: dict = {}

    @staticmethod
    def exponential_scaling(p: float, max_p: float = 1.0, alpha: float = 5.0) -> float:
        return (p / max_p) ** alpha

    def get_substitution_cost(self, src_tuple, tgt_tuple):
        key = (src_tuple, tgt_tuple)
        if key not in self.cost_cache:
            cost, best_path = self.compute_substitution_cost([src_tuple, tgt_tuple], threshold=self.threshold)
            if self.scaler == "exponential":
                cost = self.exponential_scaling(cost)
            self.cost_cache[key] = (cost, best_path)
        return self.cost_cache[key]

    def is_valid_affix(self, affix, src_dialect, tgt_dialect, affix_type):
        """
        affix: (string, 'deletion'|'insertion'|None)
        affix_type: 'prefix' or 'suffix'
        """
        if not affix or not affix[0] or not affix[1]:
            return False

        a_str, a_kind = affix
        if src_dialect == "CODA":
            src_dialect = "MSA"
        if tgt_dialect == "CODA":
            tgt_dialect = "MSA"

        affix_attached_to = ("src", src_dialect) if a_kind == "deletion" else ("tgt", tgt_dialect)
        other_dialect = tgt_dialect if affix_attached_to[0] == "src" else src_dialect
        affix_dic = self.prefixes_dict if (affix_type == "prefix") else self.suffixes_dict

        if affix_attached_to[1] == "MSA":
            return ("MSA", a_str.strip()) in affix_dic.get(other_dialect, {}).keys()
        else:
            return (affix_attached_to[1], a_str.strip()) in affix_dic.get(affix_attached_to[1], {}).keys()

    def extract_affix_spans(self, alignment):
        """
        Returns dict with prefix/suffix strings and their types ('insertion'/'deletion'/None)
        Only considers spans before the first exact match and after the last exact match.
        """
        prefix_span, suffix_span = [], []
        prefix_type, suffix_type = None, None

        alignment = alignment[::-1]
        exact_matches = [i for i, (a, b) in enumerate(alignment) if a == b]
        if not exact_matches:
            return {"prefix": "", "prefix_type": None, "suffix": "", "suffix_type": None}

        first_match = exact_matches[0]
        last_match = exact_matches[-1]

        # prefix
        for a, b in alignment[:first_match]:
            if a == -1 and b != -1:
                if prefix_type in (None, "insertion"):
                    prefix_span.append(b); prefix_type = "insertion"
                else:
                    break
            elif b == -1 and a != -1:
                if prefix_type in (None, "deletion"):
                    prefix_span.append(a); prefix_type = "deletion"
                else:
                    break
            else:
                break

        # suffix
        for a, b in alignment[last_match + 1:]:
            if a == -1 and b != -1:
                if suffix_type in (None, "insertion"):
                    suffix_span.append(b); suffix_type = "insertion"
                else:
                    break
            elif b == -1 and a != -1:
                if suffix_type in (None, "deletion"):
                    suffix_span.append(a); suffix_type = "deletion"
                else:
                    break
            else:
                break

        return {
            "prefix": "".join(prefix_span),
            "prefix_type": prefix_type,
            "suffix": "".join(suffix_span),
            "suffix_type": suffix_type,
        }

    def augmented_distance(self, input_obj, preprocess=False, verbose=False):
        # cache by 4-tuple
        key = (input_obj["source"], input_obj["target"], input_obj["source_dialect"], input_obj["target_dialect"])
        if key in self.distance_cache:
            return self.distance_cache[key]
        result = self.compute_augmented_distance(input_obj, preprocess, verbose=verbose)
        self.distance_cache[key] = result
        return result

    def compute_augmented_distance(self, input_obj, preprocess=False, verbose=False):
        src, tgt = input_obj["source"].strip(), input_obj["target"].strip()
        src_dialect, tgt_dialect = input_obj["source_dialect"], input_obj["target_dialect"]
        if src_dialect == "MSA":
            src_dialect = "CODA"
        if tgt_dialect == "MSA":
            tgt_dialect = "CODA"

        if preprocess:
            src = preprocess_text(src, normalize=True)
            tgt = preprocess_text(tgt, normalize=True)

        m, n = len(src), len(tgt)
        dp = np.full((m + 1, n + 1), float("inf"))
        backtrack = np.empty((m + 1, n + 1), dtype=object)

        dp[0][0] = 0

        # first row (insertions)
        for j in range(1, n + 1):
            if tgt[j - 1] == " ":
                insertion_cost = 0
            else:
                insertion_cost, _ = self.get_substitution_cost(("-1", src_dialect), (tgt[j - 1], tgt_dialect))
            dp[0][j] = dp[0][j - 1] + insertion_cost
            backtrack[0][j] = (0, j - 1, "insert")

        # first column (deletions)
        for i in range(1, m + 1):
            if src[i - 1] == " ":
                deletion_cost = 0
            else:
                deletion_cost, _ = self.get_substitution_cost((src[i - 1], src_dialect), ("-1", tgt_dialect))
            dp[i][0] = dp[i - 1][0] + deletion_cost
            backtrack[i][0] = (i - 1, 0, "delete")

        # DP
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                src_char, tgt_char = src[i - 1], tgt[j - 1]

                # substitution/match
                cost = 0 if src_char == tgt_char else self.get_substitution_cost((src_char, src_dialect), (tgt_char, tgt_dialect))[0]
                if dp[i - 1][j - 1] + cost < dp[i][j]:
                    dp[i][j] = dp[i - 1][j - 1] + cost
                    backtrack[i][j] = (i - 1, j - 1, "substitution")

                # insertion
                insertion_cost = 0 if (tgt_char == " ") else self.get_substitution_cost(("-1", src_dialect), (tgt_char, tgt_dialect))[0]
                if dp[i][j - 1] + insertion_cost < dp[i][j]:
                    dp[i][j] = dp[i][j - 1] + insertion_cost
                    backtrack[i][j] = (i, j - 1, "insert")

                # deletion
                deletion_cost = 0 if (src_char == " ") else self.get_substitution_cost((src_char, src_dialect), ("-1", tgt_dialect))[0]
                if dp[i - 1][j] + deletion_cost < dp[i][j]:
                    dp[i][j] = dp[i - 1][j] + deletion_cost
                    backtrack[i][j] = (i - 1, j, "delete")

        # backtrack
        i, j = m, n
        alignment = []
        while i > 0 or j > 0:
            if backtrack[i][j] is None:
                break
            prev_i, prev_j, op = backtrack[i][j]
            if op == "substitution":
                alignment.append((src[prev_i], tgt[prev_j]))
            elif op == "insert":
                alignment.append((-1, tgt[prev_j]))
            elif op == "delete":
                alignment.append((src[prev_i], -1))
            i, j = prev_i, prev_j

        total_distance = float(dp[m][n])
        denom0 = max(len(src.replace(" ", "")), len(tgt.replace(" ", "")))
        normalized_distance = total_distance / denom0 if denom0 else 0.0

        src_tokenized = src
        tgt_tokenized = tgt
        base_src_len_ns = len(src.replace(" ", ""))
        base_tgt_len_ns = len(tgt.replace(" ", ""))

        # early exit guard
        if (normalized_distance < 0.5) or (len(src.split(" ")) > 1) or (len(tgt.split(" ")) > 1):
            if verbose:
                print("handling affixes")

            affix_info = self.extract_affix_spans(alignment)
            prefix, prefix_type = affix_info["prefix"], affix_info["prefix_type"]
            suffix, suffix_type = affix_info["suffix"], affix_info["suffix_type"]

            # if prefix contains a space, recurse after trimming that whole word-span
            if prefix and (" " in prefix) and len(prefix) > 1:
                last_space_index = prefix.rfind(" ")
                trimmed = prefix[:last_space_index]
                if verbose:
                    print("prefix", trimmed)
                new_input = {
                    "source": src[last_space_index + 1:] if prefix_type == "deletion" else src,
                    "target": tgt[last_space_index + 1:] if prefix_type == "insertion" else tgt,
                    "source_dialect": input_obj["source_dialect"],
                    "target_dialect": input_obj["target_dialect"],
                }
                result = self.augmented_distance(new_input)
                if prefix_type == "deletion":
                    result["removed_src_words"] = result.get("removed_src_words", []) + [trimmed]
                elif prefix_type == "insertion":
                    result["removed_tgt_words"] = result.get("removed_tgt_words", []) + [trimmed]
                return result

            if suffix and (" " in suffix) and len(suffix) > 1:
                first_space_index = suffix.find(" ")
                trimmed = suffix[first_space_index + 1:]
                new_input = {
                    "source": src if suffix_type == "insertion" else src[: -len(trimmed) - 1],
                    "target": tgt if suffix_type == "deletion" else tgt[: -len(trimmed) - 1],
                    "source_dialect": input_obj["source_dialect"],
                    "target_dialect": input_obj["target_dialect"],
                }
                result = self.augmented_distance(new_input)
                if suffix_type == "deletion":
                    result["removed_src_words"] = result.get("removed_src_words", []) + [trimmed]
                elif suffix_type == "insertion":
                    result["removed_tgt_words"] = result.get("removed_tgt_words", []) + [trimmed]
                return result

            # validate affixes
            valid_prefix = self.is_valid_affix((prefix, prefix_type), src_dialect, tgt_dialect, "prefix") if prefix else False
            valid_suffix = self.is_valid_affix((suffix, suffix_type), src_dialect, tgt_dialect, "suffix") if suffix else False

            # adjust distance for valid affixes
            if valid_prefix:
                if prefix_type == "deletion":
                    total_distance -= dp[len(prefix)][0]  # approx: remove deletion segment
                    src_tokenized = src[:len(prefix)] + "#" + src[len(prefix):]
                    base_src_len_ns -= len(prefix.replace(" ", ""))
                elif prefix_type == "insertion":
                    total_distance -= dp[0][len(prefix)]  # approx: remove insertion segment
                    tgt_tokenized = tgt[:len(prefix)] + "#" + tgt[len(prefix):]
                    base_tgt_len_ns -= len(prefix.replace(" ", ""))

            if valid_suffix:
                if suffix_type == "deletion":
                    total_distance -= (dp[m][n] - dp[len(src) - len(suffix)][n])
                    src_tokenized = src_tokenized[:-len(suffix)] + "#" + src_tokenized[-len(suffix):]
                    base_src_len_ns -= len(suffix.replace(" ", ""))
                elif suffix_type == "insertion":
                    total_distance -= (dp[m][n] - dp[m][len(tgt) - len(suffix)])
                    tgt_tokenized = tgt_tokenized[:-len(suffix)] + "#" + tgt_tokenized[-len(suffix):]
                    base_tgt_len_ns -= len(suffix.replace(" ", ""))

            denom = max(base_src_len_ns, base_tgt_len_ns) or 1
            normalized_distance = total_distance / denom

        return {
            "alignment": alignment[::-1],
            "distance_after_removing_affixes": float(round(total_distance, 3)),
            "distance_before_removing_affixes": float(round(dp[m][n], 3)),
            "normalized_distance": float(round(normalized_distance, 3)),
            "tokenized_src": src_tokenized,
            "tokenized_tgt": tgt_tokenized,
        }

    @staticmethod
    def distance(input_obj):
        return levenshtein_distance(input_obj["source"], input_obj["target"])

# ------------- script entry -----------------
if __name__ == "__main__":
    # load prob tables & dicts first (from ./output)
    load_distance_resources()

    # read alignments and build prefixes/suffixes dictionaries
    align_path = "output/MADAR_reformatted_word_alignments.tsv"
    MADAR_26_word_alignment = pd.read_csv(align_path, sep="\t")

    # the dialect columns contain dict-like strings → actual dicts
    MADAR_26_word_alignment.loc[:, dialects_26] = MADAR_26_word_alignment.loc[:, dialects_26].map(ast.literal_eval)

    edits_dic = {
        col + "_edits": expand_alignments_with_distance(MADAR_26_word_alignment, "MSA", col)
        for col in dialects_26
    }

    # IMPORTANT: .copy() to avoid SettingWithCopyWarning
    low_edits_dic = {
        k + "_edits_low_edit": v[(v["Normalized_Edit_Distance"] <= 0.5)
                                 & (v["Frequency"] >= 1)
                                 & (v["Normalized_Edit_Distance"] > 0)].copy()
        for k, v in edits_dic.items()
    }

    # compute affix candidates (work on copies + .loc assignment)
    for key, value in low_edits_dic.items():
        if "MSA" not in key:  # skip MSA self
            dia = key.split("_")[0]
            low_edits_dic[key] = value.copy()
            low_edits_dic[key].loc[:, "Prefix"] = low_edits_dic[key].apply(extract_prefix, args=(dia,), axis=1)
            low_edits_dic[key].loc[:, "Suffix"] = low_edits_dic[key].apply(extract_suffix, args=(dia,), axis=1)

    prefixes, suffixes = {}, {}
    for key, value in low_edits_dic.items():
        if "MSA" in key:
            continue
        dia = key.split("_")[0]

        # ---- Prefix table (explicit naming to avoid KeyError: 'index')
        vc_p = value["Prefix"].dropna().value_counts()
        pf = vc_p.rename_axis("item").reset_index(name="count")   # 'item' holds the tuple
        if not pf.empty:
            # item: (('prefix', src, tgt), affix_string)
            pf["prefix"] = pf["item"].map(lambda x: x[1])
            pf["direction"] = pf["item"].map(lambda x: x[0][1])   # ('prefix', src, tgt) -> src
            pf = pf.drop(columns=["item"])
            pf.set_index(["direction", "prefix"], inplace=True)
            prefixes[dia] = pf.to_dict("index")

        # ---- Suffix table
        vc_s = value["Suffix"].dropna().value_counts()
        sf = vc_s.rename_axis("item").reset_index(name="count")
        if not sf.empty:
            # item: (('suffix', src, tgt), affix_string)
            sf["suffix"] = sf["item"].map(lambda x: x[1])
            sf["direction"] = sf["item"].map(lambda x: x[0][1])   # ('suffix', src, tgt) -> src
            sf = sf.drop(columns=["item"])
            sf.set_index(["direction", "suffix"], inplace=True)
            suffixes[dia] = sf.to_dict("index")

    # init calculator
    calc = DistanceCalculator(
        compute_substitution_cost_func=compute_substitution_cost,
        scaler="exponential",
        threshold=0,
        prefixes=prefixes,
        suffixes=suffixes,
    )

    # quick smoke test
    demo = {
        "source": "سيارة", "target": "السياره",
        "source_dialect": "MSA", "target_dialect": "BEI",
    }
    print(calc.augmented_distance(demo, preprocess=True, verbose=False))
