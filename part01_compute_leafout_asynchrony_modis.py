import os
import numpy as np
import xarray as xr
from tqdm import tqdm
from glob import glob
from osgeo import gdal
from joblib import Parallel, delayed
from numpy.lib.stride_tricks import sliding_window_view



def read_geotiff(file_path):
    """
    Read data from a GeoTIFF file.

    Parameters:
        file_path (str): Path to the GeoTIFF file.

    Returns:
        tuple: (data, geotrans, proj)
            - data (np.ndarray): Data array read from the file. For multi-band images, the shape is (bands, rows, cols).
            - geotrans (list): Affine transformation parameters in the form 
                               [top left x, pixel width, rotation, top left y, rotation, -pixel height].
            - proj (str): Projection information.
            
    In case of an error, prints an error message and returns (None, None, None).
    """
    try:
        dataset = gdal.Open(file_path)  # Open the GeoTIFF file
        if dataset is None:
            raise RuntimeError(f"Failed to open GeoTIFF file: {file_path}")

        data = dataset.ReadAsArray()  # Read the data array
        geotrans = dataset.GetGeoTransform()  # Get affine transformation parameters
        proj = dataset.GetProjection()  # Get projection information

        return data, list(geotrans), proj

    except Exception as e:
        print(f"Error in reading GeoTIFF file: {str(e)}")
        return None, None, None

    finally:
        if dataset is not None:
            dataset = None  # Release resources



def sliding_grid_partition(data: np.ndarray, target_row_blocks: int, target_col_blocks: int) -> np.ndarray:
    """
    Partition a 2D or 3D array into a grid of fixed-size blocks along its last two dimensions using a sliding window approach.
    
    For a 2D array (shape: (rows, cols)), the array is first expanded to (1, rows, cols) and then partitioned into
    target_row_blocks x target_col_blocks blocks. The output shape is:
        - For 2D input: (target_row_blocks, target_col_blocks, block_height, block_width)
        - For 3D input: (D, target_row_blocks, target_col_blocks, block_height, block_width)
    
    If the number of rows and columns can be evenly divided by the target block counts, the function uses reshape
    and transpose for efficient extraction. Otherwise, it computes a fixed block size (using ceiling division) and
    extracts blocks using sliding_window_view based on computed centers, ensuring blocks do not exceed array bounds.
    
    Parameters:
        data (np.ndarray): Input array, either a 2D array (rows, cols) or a 3D array (D, rows, cols), where D typically 
                           represents the number of slices (e.g., channels, time frames, etc.).
        target_row_blocks (int): Desired number of blocks along the row dimension.
        target_col_blocks (int): Desired number of blocks along the column dimension.
        
    Returns:
        np.ndarray: The grid-partitioned array using a sliding window approach. For 2D input, the shape is 
                    (target_row_blocks, target_col_blocks, block_height, block_width), and for 3D input, 
                    the shape is (D, target_row_blocks, target_col_blocks, block_height, block_width).
    """
    # If the input is a 2D array, expand it to 3D for uniform processing; mark to squeeze output later.
    squeeze_output = False
    if data.ndim == 2:
        data = data[np.newaxis, ...]
        squeeze_output = True
    elif data.ndim != 3:
        raise ValueError("Input data must be a 2D array (rows, cols) or a 3D array (D, rows, cols).")
    
    num_slices, orig_rows, orig_cols = data.shape
    row_scale = orig_rows / target_row_blocks
    col_scale = orig_cols / target_col_blocks

    # If the dimensions are evenly divisible, use reshape and transpose for fast extraction.
    if row_scale.is_integer() and col_scale.is_integer():
        block_row_size, block_col_size = int(row_scale), int(col_scale)
        partitioned = data.reshape(num_slices, target_row_blocks, block_row_size, target_col_blocks, block_col_size)
        partitioned = partitioned.transpose(0, 1, 3, 2, 4)
    else:
        # Compute block size with ceiling division.
        block_row_size = int(np.ceil(row_scale))
        block_col_size = int(np.ceil(col_scale))
        
        # Calculate offsets to ensure blocks do not exceed array bounds.
        offset_row_left = block_row_size // 2
        offset_row_right = block_row_size - offset_row_left
        offset_col_top = block_col_size // 2
        offset_col_bottom = block_col_size - offset_col_top
        
        # Compute the center positions for blocks along rows and columns.
        row_centers = np.linspace(offset_row_left, orig_rows - offset_row_right, target_row_blocks).round().astype(int)
        col_centers = np.linspace(offset_col_top, orig_cols - offset_col_bottom, target_col_blocks).round().astype(int)
        
        # Determine the starting indices of each block.
        row_starts = row_centers - offset_row_left
        col_starts = col_centers - offset_col_top
        
        # Use sliding_window_view to generate all possible windows of fixed block size.
        windows = sliding_window_view(data, (block_row_size, block_col_size), axis=(1, 2))
        # Create a grid to select the desired windows based on the computed starting indices.
        grid_row, grid_col = np.meshgrid(row_starts, col_starts, indexing='ij')
        partitioned = windows[:, grid_row, grid_col, :, :]
    
    if squeeze_output:
        partitioned = partitioned[0]
    return partitioned



def aggregate_pheno_to_grids(file, resolution_list=[], mask=None, pheno_name='lod', dst_dir=None):
    """
    Aggregate one yearly MODIS phenology GeoTIFF into lat/lon grids at multiple resolutions.

    For each target grid-cell size (resolution):
      1) Apply a stability mask (set unstable pixels to NaN)
      2) Partition the raster into (lat_blocks x lon_blocks) grid cells
      3) Compute within-grid mean, std, and valid-pixel fraction (area)
      4) Save one-year (time=1) NetCDF file: {pheno_name}_{mean/std/area}

    Parameters
    ----------
    file : str
        Input GeoTIFF path (one year).
    resolution_list : list[float] or None
        Target grid sizes in degrees (e.g., 0.10-1.00).
    mask : np.ndarray (bool)
        Pixel-level stability mask with the same shape as the GeoTIFF.
        True = keep pixel; False = set to NaN.
    pheno_name : str
        Phenology metric name used in variable names and output filenames (e.g., 'lod').
    dst_dir : str
        Output directory for NetCDF files.
    """

    # Extract year from filename (assumes something like: Greenup.2001.tif)
    year = os.path.basename(file).split('.')[1]
    data, geotrans, proj = read_geotiff(file)
    data[~mask] = np.nan # 去除植被变化幅度较大的像元
    cur_rows, cur_cols = data.shape


    # Loop over target grid size
    for resolution in resolution_list:
        # Convert resolution to a filename-friendly key: 0.10 -> 0p10
        resolution_key = f'{resolution:.2f}'.replace('.','p')
        output_file = f"{dst_dir}/{pheno_name}.{resolution_key}.{year}.nc"

        # Skip if already processed
        if os.path.exists(output_file): 
            continue

        # Number of grid cells:
        # lat spans 90N -> 30N (60 degrees), lon spans -180 -> 180 (360 degrees)
        exc_rows, exc_cols = int(round(60/resolution)), int(round(360/resolution))
        res = sliding_grid_partition(data, exc_rows, exc_cols).reshape(exc_rows, exc_cols, -1)

        # Add a time dimension (one yearly slice)
        res = np.expand_dims(res, axis=0)

        # Grid-cell statistics (ignore NaNs)
        res_mean = np.nanmean(res, axis=-1)
        res_std  = np.nanstd(res, axis=-1)
        res_area = np.sum(~np.isnan(res), axis=-1)/(np.ceil(cur_rows/exc_rows)*(np.ceil(cur_cols/exc_cols)))

        # Coordinates are grid-cell centers
        coords = {'time':[np.datetime64(f'{year}-01-01', 'ns')], 
                  'lat':np.arange(  90-resolution/2,  30, -resolution), 
                  'lon':np.arange(-180+resolution/2, 180,  resolution)}
        
        # Pack into a Dataset with consistent variable naming
        res_ds = xr.Dataset(data_vars={f'{pheno_name}_std' :(['time', 'lat', 'lon'], res_std), 
                            f'{pheno_name}_mean':(['time', 'lat', 'lon'], res_mean),
                            f'{pheno_name}_area':(['time', 'lat', 'lon'], res_area)},coords=coords)
        
        # Save as float32 to reduce storage size
        res_ds = res_ds.astype(np.float32)
        res_ds.to_netcdf(output_file)




if __name__ == '__main__':
    
    # Pixel-level stability mask (same shape as MODIS phenology rasters: 14400 x 86400)
    # True: keep pixel; False: set pixel to NaN
    mask = np.load('data/data_common/mask/mask_lct_2001_2023_mode_nh30.npy')

    # Target grid-cell sizes (degree)
    resolution_list = list(np.arange(0.10, 1.01, 0.05).round(4))

    # Input directory contains yearly GeoTIFFs over NH mid-high latitudes (2001–2023)
    src_dir = f'E:/DataSet_Modis/MCD12Q2.061/lod_nh30'
    # Output directory for temporary per-year NetCDFs at multiple grid sizes
    dst_dir = f'data/data_common/phenology/lod_nh30_grids'
    os.makedirs(dst_dir, exist_ok=True)

    # Expected filename pattern: {key}.{year}.tif (e.g., Greenup.2001.tif)
    files = glob(f"{src_dir}/lod.*.tif")
    # Parallel processing (limit jobs to reduce IO pressure)
    _ = Parallel(n_jobs=4)(delayed(aggregate_pheno_to_grids)(file, resolution_list, mask, 'lod', dst_dir) for file in tqdm(files))