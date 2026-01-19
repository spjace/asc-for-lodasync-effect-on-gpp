import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import zscore
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
from phenology.utils import sequence_split



def build_climate_matched_neighbors(
    climate_csv: str = "data/data_sites/site.climate.csv",
    n_neighbors: int = 200,
    cols: list[str] = ["prec", "srad", "tavg"],
):
    """
    Build climate-matched station neighborhoods for PEP725 sites.

    Reads a station-level climate table, z-score standardizes selected climate variables,
    and finds the `n_neighbors` nearest stations in the standardized climate space
    (Euclidean distance). Returns a DataFrame with neighbor station IDs and distance scores
    for each station.

    Parameters
    ----------
    climate_csv : str
        Station climate table with columns ['pep_id', 'lat', 'lon'] + `cols`.
    n_neighbors : int
        Number of nearest neighbors to retrieve for each station.
    cols : list[str]
        Climate variables used for matching (z-scored before neighbor search).

    Returns
    -------
    pd.DataFrame
        DataFrame with added columns:
        'similar_pep_ids' (neighbor pep_id array) and 'similarity_scores' (distance array).
    """
    # --- Load station-level climate table and clean ---
    df = pd.read_csv(climate_csv)[['pep_id', 'lat', 'lon', *cols]].dropna()

    # Standardize climate features (precipitation, solar radiation, temperature)
    df[cols] = df[cols].apply(zscore)

    # Fit a nearest-neighbor model to find climate-matched stations
    X = df[cols]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', metric='euclidean')
    nbrs.fit(X)

    # Query nearest neighbors for each station
    distances, indices = nbrs.kneighbors(X)

    # Store the neighbor station IDs and corresponding distances
    df['similar_pep_ids'] = [df.iloc[indices[i]]['pep_id'].values for i in range(len(df))]
    df['similarity_scores'] = [np.array(distances[i]) for i in range(len(df))]

    return df



def get_leafout_async(df: pd.DataFrame, nbr_chunk: pd.Series, key: str = "lod") -> pd.DataFrame:
    """
    Compute year-wise leaf-out timing dispersion for a climate-matched neighborhood of a target station.

    The neighborhood is selected using precomputed nearest-neighbor results in `nbr_chunk`:
    - Start with a strict distance threshold; relax it if too few species or stations remain.
    - Keep at most the top 200 most similar stations.
    - Aggregate to (year, species) means, then summarize within each year.

    Parameters
    ----------
    df : pd.DataFrame
        Phenology records containing at least ['pep_id', 'year', 'species', key].
    nbr_chunk : pd.Series
        One row from the similarity table with fields:
        ['pep_id', 'similar_pep_ids', 'similarity_scores'].
    key : str, default "lod"
        Phenology metric column name (e.g., 'lod').

    Returns
    -------
    pd.DataFrame
        Year-wise summary for the target pep_id with columns:
        ['pep_id', 'year', 'mean', 'std', 'species', 'counts'].
    """
    pep_id = nbr_chunk["pep_id"]
    similar_pep_ids = nbr_chunk["similar_pep_ids"]
    similarity_scores = nbr_chunk["similarity_scores"]

    # Start with a strict radius in the standardized 3D climate space
    mask = similarity_scores < 0.5 * np.sqrt(3)
    ids = list(similar_pep_ids[mask])

    df_sel = df[df["pep_id"].isin(ids)]
    species_num = df_sel["species"].nunique()

    # Relax the radius if the neighborhood is too small / too species-poor
    if species_num < 3 or len(ids) < 50:
        mask = similarity_scores < 1.0 * np.sqrt(3)
        ids = list(similar_pep_ids[mask])

    # Cap neighborhood size for efficiency and stability
    ids = ids[:200]

    # Aggregate within station-neighborhood: mean per (year, species)
    df_sel = df[df["pep_id"].isin(ids)]
    df_sel = df_sel.groupby(["year", "species"]).mean().reset_index()

    # Summarize within each year
    df_group_year = df_sel.groupby("year")
    result = np.full((len(df_group_year), 5), np.nan)  # year, mean, std, n_species, n_obs

    for i, (year, df_year) in enumerate(df_group_year):
        species_num = df_year["species"].nunique()
        doy = df_year[key].values

        result[i] = [
            year,
            np.mean(doy),
            np.std(doy),
            species_num,
            len(doy),
        ]

    result = pd.DataFrame(result, columns=["year", "lod_mean", "lod_std", "species", "counts"])
    result.insert(0, "pep_id", pep_id)
    return result



def main(df: pd.DataFrame, nbr_chunk: pd.DataFrame, key: str) -> list[pd.DataFrame]:
    """
    Apply `get_leafout_async` to each station in `nbr_chunk` (parallel-ready helper).

    Parameters
    ----------
    df : pd.DataFrame
        Phenology records with ['pep_id', 'year', 'species', key].
    nbr_chunk : pd.DataFrame
        Chunk of the neighbor table (one row per target station).
    key : str
        Phenology column name (e.g., 'lod').

    Returns
    -------
    list[pd.DataFrame]
        List of per-station yearly summaries.
    """
    return [get_leafout_async(df, row, key) for _, row in nbr_chunk.iterrows()]





if __name__ == "__main__":

    # --- Inputs ---
    f_lod = "data/data_sites/data_pep725/pep725_lod.csv" # species-level lod records
    f_site_climate = "data/data_sites/site.climate.csv"  # station-level climate table derived from WorldClim v2.1 (tavg/prec/srad) sampled at PEP725 site locations

    # Build climate-matched neighbors (adds: similar_pep_ids, similarity_scores)
    site_neighbors = build_climate_matched_neighbors(f_site_climate)

    df = pd.read_csv(f_lod).copy(deep=True)
    key = "lod"  # spring leaf-out date (DOY), defined as BBCH=11 (first leaf unfolded / first petiole visible)

    # Split the station list into chunks for parallel execution
    neighbors_chunks = sequence_split(site_neighbors, 256)

    # --- Parallel processing ---
    res_list = Parallel(n_jobs=24)(delayed(main)(df, nbr_chunk, key) for nbr_chunk in tqdm(neighbors_chunks))

    # Flatten and concatenate
    res_list = [item for sublist in res_list for item in sublist]
    result = pd.concat(res_list, ignore_index=True)

    # Clean dtypes for export
    result[["year", "species", "counts"]] = result[["year", "species", "counts"]].astype(int)
    output_csv = "data/data_sites/data_pep725/pep725_lod_async_climate_matched_yearly.csv"
    result.to_csv(output_csv, index=False)
