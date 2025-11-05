import pandas as pd
import numpy as np
import os
import sys

# allow relative import of utilities/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utilities.preprocess_text import preprocess_text


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

def import_caphi_table_df(path: str = "./data/CAPHI/CAPHI-table.tsv") -> pd.DataFrame:
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


if __name__ == '__main__':
    df = load_madar_coda_with_alignments()
    print(df.head())