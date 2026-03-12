import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    keys = ["age", "gender", "zip3"]

    anon_counts = anon_df.groupby(keys).size().reset_index(name="anon_count")
    aux_counts = aux_df.groupby(keys).size().reset_index(name="aux_count")

    unique_keys = anon_counts.merge(aux_counts, on=keys, how="inner")
    unique_keys = unique_keys[(unique_keys["anon_count"] == 1) & (unique_keys["aux_count"] == 1)][keys]

    matches = anon_df.merge(unique_keys, on=keys, how="inner")
    matches = matches.merge(aux_df, on=keys, how="inner")

    return matches[["anon_id", "name"]].rename(columns={"name": "matched_name"})


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    return len(matches_df) / len(anon_df)
