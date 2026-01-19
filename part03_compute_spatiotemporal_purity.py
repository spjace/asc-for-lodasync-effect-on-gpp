import os
import numpy as np
import xarray as xr
from tqdm import tqdm
from joblib  import Parallel, delayed
from phenology.utils import sliding_grid_partition



def get_mode(x: np.ndarray) -> tuple[int, int, int]:
    """
    Compute mode diagnostics for a 1D window of categorical labels.

    Zeros are treated as background/invalid and removed (x > 0).
    If the number of valid pixels is < 10, returns (0, 0, count) to flag
    insufficient support. Otherwise, returns:
      - mode_val  : most frequent positive code
      - mode_freq : frequency of mode_val
      - count     : number of valid pixels (x > 0)

    Parameters
    ----------
    x : np.ndarray
        1D array of integer-like category codes (flattened window).

    Returns
    -------
    tuple[int, int, int]
        (mode_val, mode_freq, count).
    """
    x = x[x > 0]
    count = len(x)
    if count < 10:
        return 0, 0, count

    # bincount requires non-negative integer values
    tmp = np.bincount(x)
    return int(np.argmax(tmp)), int(tmp.max()), int(count)



def centers(vmin: float, vmax: float, n: int, descending: bool = False) -> np.ndarray:
    """
    Compute equal-width bin centers between vmin and vmax.

    This helper is used to approximate coarse-grid coordinate centers when
    the coordinate spacing is regular. It splits [vmin, vmax] into n bins and
    returns the bin-center coordinates.

    Parameters
    ----------
    vmin, vmax : float
        Coordinate bounds.
    n : int
        Number of centers to generate.
    descending : bool
        If True, reverse the output (useful for north-to-south latitudes).

    Returns
    -------
    np.ndarray
        1D array of length n with bin-center coordinates.
    """
    d = (vmax - vmin) / n
    c = vmin + (np.arange(n) + 0.5) * d
    return c[::-1] if descending else c



def main(input_file, output_file, scale=480):
    """
    Aggregate a tiled categorical raster into coarse-grid mode statistics.

    This function reads one NetCDF tile containing a categorical raster (e.g., dominant
    tree species / class codes) and summarizes it on a `scale × scale` coarse grid.
    For each coarse cell, it computes:
      - mode_val  : the most frequent category code among valid pixels (>0)
      - mode_freq : the frequency of the mode_val within the coarse cell window
      - count     : the number of valid pixels (>0) within the coarse cell window

    Background / Intended usage
    ---------------------------
    The 30 m categorical species data are stored as 2°×2° tiles to reduce file size.
    A 2° span corresponds to ~480 pixels at ~500 m resolution, hence `scale=480` when
    aggregating onto a MODIS-like 500 m grid.

    Assumptions
    -----------
    - The input NetCDF contains 1D coordinate variables named 'lat' and 'lon'.
    - The tile contains exactly one categorical data layer (or, if multiple exist,
      `ds.to_array().squeeze()` must reduce to a single 2D raster).
    - Category conventions:
        > 0 : valid category/class code
        <=0 : background/invalid
    - Window coverage rule:
        coarse cells are summarized only if the fraction of valid pixels exceeds 1%
        of the window size; otherwise they are set to 0 in all outputs.

    Parameters
    ----------
    input_file : str
        Path to one input NetCDF tile.
    output_file : str
        Path to the output NetCDF file to be written.
    scale : int, default 480
        Number of coarse blocks along each tile dimension.

    Returns
    -------
    None
        Writes a NetCDF file containing mode statistics to `output_file`.
    """
    # Open one tile
    ds = xr.open_dataset(input_file)
    lat_arr = ds["lat"].values.squeeze()
    lon_arr = ds["lon"].values.squeeze()

    # Construct coarse-grid coordinate centers
    lat_list = centers(lat_arr.min(), lat_arr.max(), scale, descending=True).round(9)
    lon_list = centers(lon_arr.min(), lon_arr.max(), scale, descending=False).round(9)

    # Convert tile raster to a numpy array
    da = ds.to_array().squeeze().values

    # Partition raster into coarse windows
    da_windows = sliding_grid_partition(da, target_row_blocks=scale, target_col_blocks=scale).transpose(1, 2, 0, 3, 4).reshape(scale, scale, -1)

    # Preliminary coverage filter: only process coarse cells with >1% valid pixels (>0);
    mask = np.sum(da_windows > 0, axis=-1) > (0.01 * da_windows.shape[-1])
    res_arr = np.apply_along_axis(get_mode, axis=1, arr=da_windows[mask])

    # Write results back to the 2D coarse grid; cells failing the mask remain zeros.
    res_mask = np.zeros((mask.shape[0], mask.shape[1], res_arr.shape[-1]), dtype=np.int32)
    res_mask[mask] = res_arr.astype(np.int32)

    # Build output dataset
    var_names = ["mode_val", "mode_freq", "count"]
    res_ds = xr.Dataset(
        data_vars={name: (("lat", "lon"), res_mask[..., i]) for i, name in enumerate(var_names)},
        coords={"lat": lat_list, "lon": lon_list},
    )

    res_ds.to_netcdf(output_file)




if __name__ == '__main__':
    """
    Example workflow for computing spatiotemporal “purity” diagnostics from tiled 30 m categorical species data.

    The 30 m categorical species dataset is extremely large, so it is stored as 2°×2° tiles (2001–2022).
    For reference, a global 500 m MODIS-like grid is ~86400×43200 pixels; a 2° span corresponds to ~480
    pixels per dimension at ~500 m resolution, motivating scale=480 for tile aggregation.
    """

    src_dir = "E:/Datasets/CA_forest_lead_tree_species_grouped"
    dst_dir = "E:/Datasets/CA_forest_lead_tree_species_grouped_purity"

    # Collect all unprocessed (input, output) tile pairs
    args_list = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if not file.endswith(".nc"): continue
            input_file = os.path.join(root, file)
            output_file = input_file.replace(src_dir, dst_dir)

            # Skip tiles that already have outputs (resume-friendly)
            if os.path.exists(output_file): continue
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            args_list.append((input_file, output_file))


    scale = 480
    # Parallel execution across tiles.
    _ = Parallel(n_jobs=200)(delayed(main)(src_file, dst_file, scale) for src_file, dst_file in tqdm(args_list))