import pyEDM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from statsmodels.tsa.stattools import acf


def get_tau(df: pd.DataFrame, showPlot: bool = False) -> int:
    """
    Estimate embedding delay tau from |ACF(x)| at lags 1–3.

    Rule:
      - choose the first lag where |ACF| < 1/e; otherwise use the lag with minimum |ACF|.
    Returns NEGATIVE tau to follow pyEDM convention (past lags).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain column 'x'.
    showPlot : bool, default False
        If True, plot |ACF| (lags 1–3) and the 1/e threshold.

    Returns
    -------
    int
        Negative tau (e.g., -1, -2, -3).
    """
    # Compute |ACF| for candidate lags (1..3) only (short annual series).
    tau_list = abs(acf(df["x"], nlags=3))[1:]

    # First lag below 1/e; if none, use minimum |ACF| among lags 1..3.
    tau_min = int(np.argmin(tau_list))
    tau = next((i for i, v in enumerate(tau_list) if v < 1 / np.e), tau_min) + 1

    if showPlot:
        plt.figure(figsize=(8, 4))
        plt.stem(range(len(tau_list)), tau_list)
        plt.axhline(y=1 / np.e, color="r", linestyle="--")  # heuristic threshold
        plt.xlabel("Lag")
        plt.ylabel("|ACF|")
        plt.show()

    # pyEDM uses negative tau to indicate past lags.
    return -int(tau)


def get_e(df: pd.DataFrame, tau: int, length: int) -> int:
    """
    Select embedding dimension E by maximizing rho from pyEDM.EmbedDimension.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns 'x' and 'y'.
    tau : int
        Embedding delay (negative; from `get_tau`).
    length : int
        Series length after preprocessing (pyEDM uses 1-based indices).

    Returns
    -------
    int
        Best E (1..4).
    """
    out = pyEDM.EmbedDimension(
        dataFrame=df,
        columns="x",              # manifold variable
        target="y",               # predicted variable for rho
        lib=f"1 {length}",        # library index range (1-based)
        pred=f"1 {length}",       # prediction index range (1-based)
        maxE=4,
        tau=tau,
        showPlot=False,
    )

    # Because the row index is 0-based, we add 1 to convert it to the actual embedding dimension E (which starts at 1).
    return int(out["rho"].idxmax() + 1)


def ccm_single(data, var_list=("y", "x"), showPlot: bool = False):
    """
    Run CCM for one sample in the direction: x xmap y.

    Steps:
      1) drop NaNs and z-score both variables
      2) choose tau from |ACF(x)| (lags 1–3)
      3) choose E by maximizing rho from EmbedDimension
      4) run pyEDM.CCM and return (res, tau, E)

    Parameters
    ----------
    data : array-like, shape (n_time, 2)
        Two-variable time series.
    var_list : tuple[str, str], default ("y","x")
        Column names for the two variables. Must produce 'x' and 'y' columns.
    showPlot : bool, default False
        If True, plot CCM rho vs library size.

    Returns
    -------
    tuple
        (res, tau, E) where res is the CCM output table.
    """
    # Build DataFrame; keep only complete rows.
    df = pd.DataFrame(data=data, columns=list(var_list)).dropna()
    df = df.apply(zscore).dropna()  # standardize (mean=0, std=1)
    length = len(df)

    # Auto-select embedding hyperparameters.
    tau = get_tau(df)
    E = get_e(df, tau=tau, length=length)

    # CCM: test whether x-manifold predicts y (x xmap y).
    res = pyEDM.CCM(
        dataFrame=df,
        E=E,
        columns="x",
        target="y",
        libSizes=f"{E+2} {length} 1",  # start at E+2, end at full length, step=1
        sample=30,                    # number of random subsamples per libSize
        showPlot=showPlot,
        tau=tau,
        seed=1,                       # reproducible CCM curves
    )
    return res, tau, E



if __name__ == "__main__":
    """
    Example usage: run CCM across many grid cells / samples and save results in chunks.

    Input
    -----
    - A preprocessed 3D array saved in NPZ:
        data_arr.shape = (n_samples, n_time, 2)
      where the last dimension stores two variables in the order [y, x].

    Output
    ------
    - Chunked pickle files:
        result/part3_causality/ccm_auto_tau_e/ccm_part_XX.pkl
      Each file stores a list of per-sample outputs: (res, tau, E).

    Notes
    -----
    - CCM is computationally expensive, so we split samples into chunks to:
        (i) improve parallel throughput, and
        (ii) allow stopping/resuming without losing finished parts.
    """

    import os
    import pickle
    from joblib import Parallel, delayed
    from phenology.utils import sequence_split

    # Preprocessed array: (n_samples, n_time, 2) where features are [y, x].
    data_arr = np.load("data/data_delta_lst_lud_thr13_mask.npz")["data"]
    print(f'total len: {len(data_arr)}')

    # Quick sanity check (with plotting).
    res, tau, e = ccm_single(data_arr[0], showPlot=True)

    dst_dir = f'result/part3_causality/ccm_auto_tau_e'
    os.makedirs(dst_dir, exist_ok=True)

    # Split into chunks because CCM is very time-consuming (easy to pause/resume).
    data_list = sequence_split(data_arr, 100)
    for i in range(len(data_list)):
        data_list_i = data_list[i]
        output_file_i = os.path.join(dst_dir, f'ccm_part_{str(i).zfill(2)}.pkl')
        print(f'part{i} {output_file_i} start!')

        # Skip if this chunk is already computed (resume-friendly).
        if os.path.exists(output_file_i): continue
        
        # parallel CCM over samples in this chunk; tqdm shows progress within the chunk
        res = Parallel(n_jobs=-1)(delayed(ccm_single)(data_i) for data_i in data_list_i)
        pickle.dump(res, open(output_file_i, 'wb'))