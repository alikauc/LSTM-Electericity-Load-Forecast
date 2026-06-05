#!/usr/bin/env python3
"""
Standalone ARIMA Testing and Hyperparameter Optimization Tool
Evaluates or optimizes the ARIMA baseline model independently of the LSTM.
"""

import os
import argparse
import logging
import math
import warnings
import time
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

from data_loader import prepare_data, DEFAULT_FEATURES
from arima_model import process_single_forecast_arima, DEFAULT_ORDER, FALLBACK_ORDER, HISTORY_HOURS, FORECAST_HORIZON
from log_config import setup_logging

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone ARIMA Testing and Hyperparameter Optimization Tool")
    parser.add_argument(
        "--mode", type=str, choices=["evaluate", "optimize"], default="evaluate",
        help="Mode of operation: 'evaluate' to run a single configuration, 'optimize' to perform a parameter grid search."
    )
    parser.add_argument(
        "--csv-file", type=str,
        default="An-LSTM-Based-Approach-to-Day-Ahead-Electricity-Load-Forecasting-main/data_complete.csv",
        help="Path to the dataset CSV file."
    )
    parser.add_argument(
        "--split", type=str, choices=["val", "test"], default="val",
        help="Dataset split to evaluate on. 'val' is recommended for parameter optimization. Default: val."
    )
    parser.add_argument(
        "--stride", type=int, default=24,
        help="Stride (step size) through the dataset sequence indices. A stride of 24 evaluates daily (extremely fast). Default: 24."
    )
    parser.add_argument(
        "--p", type=int, default=DEFAULT_ORDER[0], help="ARIMA autoregressive order p. Default: 2."
    )
    parser.add_argument(
        "--d", type=int, default=DEFAULT_ORDER[1], help="ARIMA differencing order d. Default: 1."
    )
    parser.add_argument(
        "--q", type=int, default=DEFAULT_ORDER[2], help="ARIMA moving average order q. Default: 2."
    )
    parser.add_argument(
        "--fallback-p", type=int, default=FALLBACK_ORDER[0], help="Fallback ARIMA order p. Default: 1."
    )
    parser.add_argument(
        "--fallback-d", type=int, default=FALLBACK_ORDER[1], help="Fallback ARIMA order d. Default: 1."
    )
    parser.add_argument(
        "--fallback-q", type=int, default=FALLBACK_ORDER[2], help="Fallback ARIMA order q. Default: 1."
    )
    parser.add_argument(
        "--history", type=int, default=HISTORY_HOURS, help="Historical context window size in hours. Default: 168."
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1, help="Number of parallel jobs to run. Default: -1 (all CPU cores)."
    )
    parser.add_argument(
        "--array-index", type=int, default=None,
        help="Deterministic index of the parameter combination to evaluate (for SLURM Array Jobs). If provided, overrides single p,d,q parameters."
    )
    parser.add_argument(
        "--out-csv", type=str, default="arima_opt_results.csv",
        help="Output path for grid search CSV results. Default: arima_opt_results.csv."
    )
    return parser.parse_args()


def get_all_combinations() -> List[Tuple[Tuple[int, int, int], int]]:
    """
    Returns the deterministic grid of ARIMA orders and history window configurations.
    """
    p_choices = [0, 1, 2]
    d_choices = [0, 1]
    q_choices = [0, 1, 2]
    history_choices = [72, 168, 336]
    
    param_grid = []
    for h in history_choices:
        for p in p_choices:
            for d in d_choices:
                for q in q_choices:
                    if p == 0 and d == 0 and q == 0:
                        continue
                    param_grid.append(((p, d, q), h))
    return param_grid


def get_split_indices(
    df_len: int, input_len: int, output_len: int, split: str, stride: int
) -> Tuple[List[int], pd.Index]:
    """
    Computes chronological start indices for the chosen dataset split and step stride.
    """
    dataset_length = df_len - input_len - output_len
    train_size = int(dataset_length * 0.70)
    val_size = int(dataset_length * 0.15)

    if split == "val":
        start_idx = train_size
        end_idx = train_size + val_size
    else:  # split == "test"
        start_idx = train_size + val_size
        end_idx = dataset_length

    # Apply the stride
    indices = list(range(start_idx, end_idx, stride))
    return indices


def run_evaluation(
    df_unscaled: pd.DataFrame,
    full_timestamps: pd.Index,
    indices: List[int],
    order: Tuple[int, int, int],
    fallback_order: Tuple[int, int, int],
    history_hours: int,
    n_jobs: int
) -> Tuple[float, float, float]:
    """
    Executes parallel ARIMA predictions over the dataset indices and calculates overall metrics.
    """
    # process_single_forecast_arima expects input_len = 24 (context window before target prediction)
    input_len = 24
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_forecast_arima)(
            idx, df_unscaled, full_timestamps, input_len,
            order=order, fallback_order=fallback_order, history_hours=history_hours
        )
        for idx in indices
    )
    
    # Filter skipped and None results
    valid_results = [r for r in results if r is not None]
    
    if len(valid_results) == 0:
        return float('nan'), float('nan'), float('nan')
        
    mae = np.mean([r["MAE"] for r in valid_results])
    rmse = np.mean([r["RMSE"] for r in valid_results])
    mape = np.mean([r["MAPE"] for r in valid_results])
    
    return mae, rmse, mape


def main():
    setup_logging()
    args = parse_args()
    
    csv_file = args.csv_file
    if not os.path.exists(csv_file):
        # Fallback to current directory check
        csv_file = "data_complete.csv"
        
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Could not locate dataset CSV file at {args.csv_file} or in current folder.")

    logger.info("====================================================================")
    logger.info("STANDALONE ARIMA forecaster and optimizer tool")
    logger.info("====================================================================")
    logger.info("Loading data from: %s", csv_file)
    
    # We load df_unscaled using the indexing pattern of prepare_data
    df_unscaled = pd.read_csv(csv_file, parse_dates=["Datetime"], index_col="Datetime")
    full_timestamps = df_unscaled.index
    
    # LSTM dataset configurations (needed for structural matching)
    input_len = 24
    output_len = 24
    
    indices = get_split_indices(len(df_unscaled), input_len, output_len, args.split, args.stride)
    
    logger.info("Dataset split:    %s", args.split.upper())
    logger.info("Index stride:     %d (evaluating every %dth sequence window)", args.stride, args.stride)
    logger.info("Total sequences:  %d sequences out of %d total windows", len(indices), len(df_unscaled) - input_len - output_len)
    logger.info("Parallel workers: %d (All cores if -1)", args.n_jobs)
    logger.info("====================================================================")

    if args.array_index is not None:
        combinations = get_all_combinations()
        if args.array_index < 0 or args.array_index >= len(combinations):
            raise ValueError(f"Array index {args.array_index} is out of bounds (0 to {len(combinations)-1}).")
            
        order, h = combinations[args.array_index]
        fallback_order = (args.fallback_p, args.fallback_d, args.fallback_q)
        
        logger.info("Evaluating Combination Index %d: %s | History: %d hours", args.array_index, order, h)
        logger.info("Running ARIMA evaluation ...")
        
        start_time = time.time()
        mae, rmse, mape = run_evaluation(
            df_unscaled, full_timestamps, indices, order, fallback_order, h, args.n_jobs
        )
        elapsed = time.time() - start_time
        
        logger.info("Evaluation Complete in %.2fs!", elapsed)
        
        # Save results to array result file
        res_file = f"arima_res_{args.array_index}.csv"
        res_df = pd.DataFrame([{
            "Index": args.array_index,
            "Order": f"ARIMA{order}",
            "p": order[0],
            "d": order[1],
            "q": order[2],
            "History_Hours": h,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "Eval_Time_Sec": round(elapsed, 2)
        }])
        res_df.to_csv(res_file, index=False)
        logger.info("Results successfully written to %s", res_file)
        return

    if args.mode == "evaluate":
        order = (args.p, args.d, args.q)
        fallback_order = (args.fallback_p, args.fallback_d, args.fallback_q)
        history_hours = args.history
        
        logger.info("Evaluating Configuration:")
        logger.info("  Primary ARIMA Order:  %s", order)
        logger.info("  Fallback ARIMA Order: %s", fallback_order)
        logger.info("  History hours:        %d hours (%.1f days)", history_hours, history_hours/24)
        logger.info("Running ARIMA evaluation ...")
        
        start_time = time.time()
        mae, rmse, mape = run_evaluation(
            df_unscaled, full_timestamps, indices, order, fallback_order, history_hours, args.n_jobs
        )
        elapsed = time.time() - start_time
        
        logger.info("Evaluation Complete in %.2fs!", elapsed)
        logger.info("--------------------------------------------------------------------")
        logger.info("Results (%s split, stride=%d):", args.split.upper(), args.stride)
        logger.info("  Mean Absolute Error (MAE):       %.2f MW", mae)
        logger.info("  Root Mean Squared Error (RMSE):   %.2f MW", rmse)
        logger.info("  Mean Absolute Pct Error (MAPE):  %.2f%%", mape)
        logger.info("--------------------------------------------------------------------")
        
    elif args.mode == "optimize":
        # Define grid search parameter space
        p_choices = [0, 1, 2]
        d_choices = [0, 1]
        q_choices = [0, 1, 2]
        history_choices = [72, 168, 336]  # 3 days, 7 days, 14 days
        
        fallback_order = (args.fallback_p, args.fallback_d, args.fallback_q)
        
        # Build list of combinations
        param_grid = []
        for h in history_choices:
            for p in p_choices:
                for d in d_choices:
                    for q in q_choices:
                        # Avoid p=0, d=0, q=0 as it is invalid/empty
                        if p == 0 and d == 0 and q == 0:
                            continue
                        param_grid.append(((p, d, q), h))
                        
        total_runs = len(param_grid)
        logger.info("Starting Hyperparameter Grid Search (Total configurations to check: %d)", total_runs)
        logger.info("Sweep Grid: p=%s, d=%s, q=%s, history=%s", p_choices, d_choices, q_choices, history_choices)
        logger.info("--------------------------------------------------------------------")
        
        leaderboard = []
        
        for idx, (order, h) in enumerate(param_grid):
            logger.info("[%d/%d] Testing ARIMA%s | History: %d hours ...", idx+1, total_runs, order, h)
            t_start = time.time()
            mae, rmse, mape = run_evaluation(
                df_unscaled, full_timestamps, indices, order, fallback_order, h, args.n_jobs
            )
            t_elapsed = time.time() - t_start
            
            if not math.isnan(mae):
                logger.info("MAE: %.2f | MAPE: %.2f%% | Time: %.1fs", mae, mape, t_elapsed)
                leaderboard.append({
                    "Order": f"ARIMA{order}",
                    "p": order[0],
                    "d": order[1],
                    "q": order[2],
                    "History_Hours": h,
                    "MAE": mae,
                    "RMSE": rmse,
                    "MAPE": mape,
                    "Eval_Time_Sec": round(t_elapsed, 2)
                })
            else:
                logger.warning("FAILED/SKIPPED")
                
        # Sort leaderboard by MAE
        results_df = pd.DataFrame(leaderboard)
        if not results_df.empty:
            results_df = results_df.sort_values(by="MAE").reset_index(drop=True)
            results_df.to_csv(args.out_csv, index=False)
            
            logger.info("====================================================================")
            logger.info("ARIMA OPTIMIZATION LEADERBOARD")
            logger.info("====================================================================")
            logger.info("\n%s", results_df[["Order", "History_Hours", "MAE", "RMSE", "MAPE", "Eval_Time_Sec"]].to_string())
            logger.info("====================================================================")
            logger.info("Full results saved to: %s", args.out_csv)
            logger.info("Best Configuration: %s with %s hours history!", results_df['Order'].iloc[0], results_df['History_Hours'].iloc[0])
        else:
            logger.warning("No valid configurations succeeded.")


if __name__ == "__main__":
    main()
