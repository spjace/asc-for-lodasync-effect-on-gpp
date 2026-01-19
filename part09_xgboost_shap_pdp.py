import os
import shap
import joblib
import optuna
import xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.text as mtext
from xgboost import XGBRegressor
from scipy.stats import zscore
from phenology.utils import remove_outliers, fit_ols_model
from phenology.plot import plt_init_start, plt_init_end
from sklearn.metrics import mean_squared_error
from scipy.interpolate import splrep, splev
from sklearn.metrics import r2_score



def fit_spline_ci(x, y, smoothing=10, knots=None, num_points=100, num_bootstrap=500):
    """
    Fit a cubic spline to (x, y) and estimate a 95% confidence interval via residual bootstrap.

    This function:
      1) Fits a cubic spline using `scipy.interpolate.splrep`.
      2) Computes fitted values at the observed x, obtains residuals, and evaluates R².
      3) Performs residual bootstrap: resamples residuals with replacement, constructs bootstrap y,
         refits the spline, and predicts on a dense x grid.
      4) Derives the 95% CI as the 2.5th and 97.5th percentiles of bootstrap predictions.

    Parameters
    ----------
    x : array-like
        1D array of predictor values.
    y : array-like
        1D array of response values (same length as x).
    smoothing : float, default 10
        Smoothing factor passed to `splrep` (larger => smoother curve).
    knots : array-like or None, default None
        Optional interior knots passed to `splrep` via `t`.
    num_points : int, default 100
        Number of x grid points used to evaluate the fitted spline and CI.
    num_bootstrap : int, default 500
        Number of bootstrap replicates used to estimate uncertainty.

    Returns
    -------
    x_new : np.ndarray
        Monotonic grid spanning [min(x), max(x)] with length `num_points`.
    y_fit : np.ndarray
        Spline predictions on `x_new`.
    ci_lower : np.ndarray
        Lower bound of the 95% CI on `x_new`.
    ci_upper : np.ndarray
        Upper bound of the 95% CI on `x_new`.
    r2 : float
        Coefficient of determination computed on the original data x.
    """
    tck = splrep(x, y, s=smoothing, t=knots)
    x_new = np.linspace(np.min(x), np.max(x), num_points)
    y_fit = splev(x_new, tck, der=0)

    # compute fitted values at observed x and residuals
    y_fit_at_data = splev(x, tck, der=0)
    residuals = y - y_fit_at_data
    r2 = r2_score(y, y_fit_at_data)

    # store bootstrap spline predictions on x_new
    bootstrap_predictions = np.zeros((num_bootstrap, len(x_new)))

    # residual bootstrap: resample residuals and refit spline
    for i in range(num_bootstrap):
        sampled_residuals = np.random.choice(residuals, size=len(y), replace=True)
        y_bootstrap = y_fit_at_data + sampled_residuals
        try:
            tck_bootstrap = splrep(x, y_bootstrap, s=smoothing, t=knots)
            bootstrap_predictions[i, :] = splev(x_new, tck_bootstrap, der=0)
        except Exception as err:
            bootstrap_predictions[i, :] = np.nan

    # 95% CI from bootstrap quantiles
    ci_lower = np.nanpercentile(bootstrap_predictions, 2.5, axis=0)
    ci_upper = np.nanpercentile(bootstrap_predictions, 97.5, axis=0)

    return x_new, y_fit, ci_lower, ci_upper, r2



def plt_importance_shap(shap_values, title=None, width=3.33, height=3.33):
    """
    Plot SHAP feature importance as a bar chart.

    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP values object returned by a SHAP explainer (e.g., TreeExplainer / GPUTreeExplainer).
    title : str or None, default None
        Optional plot title.
    width, height : float, default 3.33
        Figure size in inches.

    Returns
    -------
    None
        Displays the plot.
    """

    # Initialize figure
    fig = plt_init_start(width=width, height=height)
    ax = fig.axes[0]

    # SHAP built-in importance bar plot (top `max_display` features)
    shap.plots.bar(shap_values, max_display=40, ax=ax, show=False)

    # Force smaller font sizes for publication-style compact plots
    for text_obj in fig.findobj(match=mtext.Text): text_obj.set_fontsize(5.5)
    ax.tick_params(labelsize=6.5)
    ax.set_xlabel(ax.get_xlabel(), fontsize=6.5)
    ax.grid(False)
    if title is not None:
        plt.title(title, fontsize=6.5)
    plt_init_end()
    plt.show()


def plt_pdp(df_pdp, y_name='values', x_name='data', k=1, method=None, m2_degree=1, show_original=False, lcolor='blue', ax=None, **kwargs):
    """
    Plot a partial dependence curve with a fitted trend and uncertainty band.

    This function expects a dataframe containing the x grid (`x_name`) and the
    corresponding PDP response (`y_name`). It can fit:
      - A spline curve with bootstrap CI (default), or
      - An OLS-based curve via `fit_ols_model` when method == 'm2'.

    Parameters
    ----------
    df_pdp : pd.DataFrame
        Partial dependence results containing columns `x_name` and `y_name`.
    y_name : str, default 'values'
        Column name for PDP response values.
    x_name : str, default 'data'
        Column name for PDP x-axis values.
    k : float, default 1
        Multiplier controlling the width of the shaded uncertainty band.
    method : str or None, default None
        If 'm2', uses `fit_ols_model`; otherwise uses `fit_spline_ci`.
    m2_degree : int, default 1
        Polynomial degree used by `fit_ols_model` when method == 'm2'.
    show_original : bool, default False
        If True, plots original (x, y) points/line as a thin black curve.
    lcolor : str, default 'blue'
        Line color for fitted curve.
    ax : matplotlib.axes.Axes or None, default None
        Target axis to plot on (must be provided by the caller in your current usage).
    **kwargs
        Passed through to `fit_spline_ci` or `fit_ols_model`.

    Returns
    -------
    None
        Draws on the provided axis.
    """
    x = df_pdp[x_name].values
    y = df_pdp[y_name].values
    
    # Optionally overlay original PDP values
    if show_original:
        ax.plot(x, y, color='k', linewidth=0.5, zorder=4, alpha=0.6)

    # Fit curve + CI
    if method == 'm2':
        x_range, y_range, coef, p_val, intercept, r2, model, ci_lower, ci_upper = fit_ols_model(x, y, m2_degree, **kwargs)
    else:
        x_range, y_range, ci_lower, ci_upper, r2 = fit_spline_ci(x, y, **kwargs)
    # Fitted curve
    ax.plot(x_range, y_range, color=lcolor, linewidth=0.9, zorder=3)
    ax.fill_between(x_range, y_range - k*(y_range - ci_lower), y_range + k*(ci_upper -y_range), color='gray', alpha=0.3, zorder=1, linewidth=0.3)
    ax.text(0.05, 0.90, f'$R^2$={r2:.2f}', transform=plt.gca().transAxes, fontstyle='italic', fontsize=5, color=lcolor, ha='left', va='top', )
    ax.axhline(0, color='k', linewidth=0.4, zorder=0, linestyle='--')
    

def plt_shap(df:pd.DataFrame, y_name:str, x_name:str, bins=10, show_scatter=True, show_box=True, color_box='steelblue', ax1=None):
    """
    Plot a SHAP-style relationship figure: scatter + binned boxplots of y vs x.

    This helper is used to visualize how a response (e.g., SHAP value or model output)
    varies with a predictor by:
      - optional scatter plot of all points
      - optional boxplots of y within bins of x (pd.cut)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing columns y_name and x_name.
    y_name : str
        Column name for y values (e.g., SHAP values for a feature).
    x_name : str
        Column name for x values (feature values).
    bins : int, default 10
        Number of bins for discretizing x.
    show_scatter : bool, default True
        Whether to draw the raw scatter points.
    show_box : bool, default True
        Whether to draw binned boxplots.
    color_box : str or iterable, default 'steelblue'
        Box face color(s). (If iterable, one color per box.)
    ax1 : matplotlib.axes.Axes or None, default None
        Axis to plot on. If None, creates a new figure/axis.

    Returns
    -------
    None
        Draws on the provided axis.
    """
    if ax1 is None:
        fig = plt_init_start()
        ax1 = fig.axes[0]

    x_level = f'{x_name} level'
    df[x_level] = pd.cut(df[x_name], bins=bins)
    x_min, x_max = df[x_name].min(), df[x_name].max()
    step = (x_max - x_min) / bins

    if show_scatter:        
        ax1.scatter(x=x_name, y=y_name, rasterized=True, data=df, color='gray', edgecolor='white', alpha=0.2, s=5, linewidths=0.3, label=None)

    if show_box: # draw boxplots by x-bins
        groups = [grp[y_name].dropna().values for _, grp in df.groupby(x_level, observed=False)]        # y arrays per bin
        midpoints = [(interval.left + interval.right) / 2  for interval in df[x_level].cat.categories]  # bin centers

        box_dict1 = dict(color='k', linewidth=0.3)
        ax_box = ax1.boxplot(
            groups,
            positions=midpoints,
            widths=0.8 * step,
            patch_artist=True,
            showfliers=False,
            showmeans=False,           
            boxprops=box_dict1.copy(),
            whiskerprops=box_dict1.copy(),
            capprops=box_dict1.copy(),
            medianprops=box_dict1.copy(),
        )

        # Apply box fill colors
        for patch, col in zip(ax_box['boxes'], color_box):
            patch.set_facecolor(col)
            patch.set_alpha(1.0)

    # Reference line at y=0
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    xticks = np.arange(x_min*0.99, x_max*1.01, (x_max-x_min)/4).round(1)
    ax1.set_xticks(xticks, xticks)
    ax1.set_xlim(x_min-0.4*step, x_max+0.4*step)



def search_best_params(df, y_name, x_names, n_trials):
    """
    Search best XGBoost hyperparameters using Optuna (80/20 split + early stopping).

    The dataframe is shuffled, split into training and validation subsets (first 80% / last 20%),
    and Optuna minimizes validation RMSE. Early stopping is used during boosting, and the best
    number of estimators is saved to the trial attributes.

    Parameters
    ----------
    df : pd.DataFrame
        Input samples containing y_name and x_names columns (already preprocessed if desired).
    y_name : str
        Target variable name.
    x_names : list[str]
        Predictor variable names.
    n_trials : int
        Number of Optuna trials.

    Returns
    -------
    dict
        Best hyperparameter dict, including 'n_estimators' updated to the best iteration found by early stopping.
    """
    # Shuffle rows for a random train/val split
    df = df.sample(frac=1).reset_index(drop=True)

    X = df[x_names].values
    y = df[y_name].values

    # Simple hold-out split (80% train / 20% validation)
    split = int(len(df) * 0.8)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    def objective(trial):
        # Parameter search space
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "min_child_weight": trial.suggest_int("min_child_weight", 4, 20),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 30.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        }

        model = XGBRegressor(
            **params,
            n_estimators=10000,
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            device="cuda",
            verbosity=0,
            callbacks=[xgb.callback.EarlyStopping(rounds=100, save_best=True)],
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        yhat = model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, yhat)))

        best_iter = getattr(model, "best_iteration", None)
        if best_iter is not None:
            trial.set_user_attr("best_n_estimators", int(best_iter) + 1)
        return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = dict(study.best_params)
    best_params["n_estimators"] = int(study.best_trial.user_attrs.get("best_n_estimators", 10000))
    return best_params




def xgboost_shap(y_name, x_names, file_csv, dst_dir, plot=False, overwrite=False):
    """
    Train an XGBoost regression model, compute SHAP values, and cache results to disk.

    Workflow:
      1) Load CSV and keep (y_name + x_names).
      2) Drop NaNs, remove outliers, z-score standardize, and drop remaining NaNs.
      3) Run Optuna hyperparameter search to obtain best XGBoost params.
      4) Fit final XGBRegressor on all samples (GPU histogram).
      5) Compute SHAP values using `shap.GPUTreeExplainer`.
      6) Save (model, explainer, shap_values, df, r2) via joblib for reuse.

    Parameters
    ----------
    y_name : str
        Target variable column name.
    x_names : list[str]
        Predictor variable column names.
    file_csv : str
        Path to input CSV containing the required columns.
    dst_dir : str
        Output directory to save/load cached SHAP results.
    plot : bool, default False
        If True, plots SHAP feature importance bar plot.
    overwrite : bool, default False
        If True, recomputes and overwrites cached results even if they exist.

    Returns
    -------
    model : xgboost.XGBRegressor
        Trained XGBoost regressor.
    explainer : shap.explainers._tree.TreeExplainer
        SHAP explainer used to compute SHAP values.
    shap_values : shap.Explanation
        SHAP values object for all samples.
    df : pd.DataFrame
        Preprocessed dataframe used for model fitting.
    r2 : float
        Training R² (model.score on the fitted data).
    """
    output_file = f"{dst_dir}/shap.{y_name}.{'.'.join(x_names)}.pkl"
    
    # If cache is missing (or overwrite=True), recompute everything
    if not os.path.exists(output_file) or overwrite:
        df = pd.read_csv(file_csv)

        # Subset columns and preprocess
        df = df[[y_name, *x_names]].dropna()
        df = df.apply(remove_outliers).dropna()
        df = df.apply(zscore)
        df = df.dropna()

        # Hyperparameter tuning
        best_params = search_best_params(df, y_name, x_names, 100)
        y = df[y_name]
        x = df[x_names]

        # Fit final model and compute SHAP values (GPU)
        model = xgboost.XGBRegressor(**best_params, tree_method="hist", device="cuda").fit(x, y)
        explainer = shap.GPUTreeExplainer(model)
        shap_values = explainer(x, check_additivity=False)
        r2 = model.score(x, y)

        # Cache everything for reuse
        result_dict = {
        'model': model,
        'explainer': explainer,
        'shap_values': shap_values,
        'df': df,
        'r2': r2
        }

        joblib.dump(result_dict, output_file)

    else:
        # Load cached results
        result_dict = joblib.load(output_file)
        model, explainer, shap_values, df, r2 = result_dict['model'], result_dict['explainer'], result_dict['shap_values'], result_dict['df'], result_dict['r2']
    
    if plot:
        plt_importance_shap(shap_values)

    return model, explainer, shap_values, df, r2



if __name__ == '__main__':
    """
    Example usage:
    
    y_name = 'async_trend' # Trend of leaf-out asynchrony
    x_names = [
        'ts',     # Temperature-sensitivity
        'tssd',   # Temperature-sensitivity SD
        'soc',    # Soil organic carbon
        'socsd',  # Soil organic carbon SD
        'swr',    # Within-spring warming rate
        'swrsd',  # Within-spring warming rate SD
        'fvc',    # Fractional vegetation cover
        'fvcsd',  # Fractional vegetation cover SD
        'mst',    # Mean spring temperature
        'stsd',   # Spring temperature SD
        'msp',    # Mean spring precipitation
        'spsd',   # Spring precipitation SD
        'msr',    # Mean spring radiation
        'srsd',   # Spring radiation SD
        'elev',   # Elevation
        'elevsd', # Elevation SD
        'fa',     # Forest age
        'fasd',   # Forest age SD
        'n',      # Soil total nitrogen
        'nsd'     # Soil total nitrogen SD
    ]

    # Optional: sort predictors to keep a consistent order across runs
    x_names = sorted(x_names)
    var_list = [y_name, *x_names]

    # Input data
    file_csv='data/data_0p10/data_part2.csv', 
    dst_dir='result/part4_xgboost&shap/model'
    os.makedirs(dst_dir, exist_ok=True)

    # Run training + SHAP
    model, explainer, shap_values, df, r2 = xgboost_shap(y_name, x_names, file_csv, dst_dir)
 
    """

    pass