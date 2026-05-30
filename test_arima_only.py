#!/usr/bin/env python3
"""
Standalone ARIMA Testing and Hyperparameter Optimization Tool
Evaluates or optimizes the ARIMA baseline model independently of the LSTM.
"""

import os
import argparse
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

warnings.filterwarnings("ignore")


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
    args = parse_args()
    
    csv_file = args.csv_file
    if not os.path.exists(csv_file):
        # Fallback to current directory check
        csv_file = "data_complete.csv"
        
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Could not locate dataset CSV file at {args.csv_file} or in current folder.")

    print("====================================================================")
    print("📈 STANDALONE ARIMA forecaster and optimizer tool")
    print("====================================================================")
    print(f"Loading data from: {csv_file}")
    
    # We load df_unscaled using the indexing pattern of prepare_data
    df_unscaled = pd.read_csv(csv_file, parse_dates=["Datetime"], index_col="Datetime")
    full_timestamps = df_unscaled.index
    
    # LSTM dataset configurations (needed for structural matching)
    input_len = 24
    output_len = 24
    
    indices = get_split_indices(len(df_unscaled), input_len, output_len, args.split, args.stride)
    
    print(f"Dataset split:    {args.split.upper()}")
    print(f"Index stride:     {args.stride} (evaluating every {args.stride}th sequence window)")
    print(f"Total sequences:  {len(indices)} sequences out of {len(df_unscaled) - input_len - output_len} total windows")
    print(f"Parallel workers: {args.n_jobs} (All cores if -1)")
    print("====================================================================")

    if args.array_index is not None:
        combinations = get_all_combinations()
        if args.array_index < 0 or args.array_index >= len(combinations):
            raise ValueError(f"Array index {args.array_index} is out of bounds (0 to {len(combinations)-1}).")
            
        order, h = combinations[args.array_index]
        fallback_order = (args.fallback_p, args.fallback_d, args.fallback_q)
        
        print(f"Evaluating Combination Index {args.array_index}: {order} | History: {h} hours")
        print("\nRunning ARIMA evaluation ...")
        
        start_time = time.time()
        mae, rmse, mape = run_evaluation(
            df_unscaled, full_timestamps, indices, order, fallback_order, h, args.n_jobs
        )
        elapsed = time.time() - start_time
        
        print("\n✨ Evaluation Complete in {:.2f}s!".format(elapsed))
        
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
        print(f"Results successfully written to {res_file}")
        return

    if args.mode == "evaluate":
        order = (args.p, args.d, args.q)
        fallback_order = (args.fallback_p, args.fallback_d, args.fallback_q)
        history_hours = args.history
        
        print(f"Evaluating Configuration:")
        print(f"  • Primary ARIMA Order:  {order}")
        print(f"  • Fallback ARIMA Order: {fallback_order}")
        print(f"  • History hours:        {history_hours} hours ({history_hours/24:.1f} days)")
        print("\nRunning ARIMA evaluation ...")
        
        start_time = time.time()
        mae, rmse, mape = run_evaluation(
            df_unscaled, full_timestamps, indices, order, fallback_order, history_hours, args.n_jobs
        )
        elapsed = time.time() - start_time
        
        print("\n✨ Evaluation Complete in {:.2f}s!".format(elapsed))
        print("--------------------------------------------------------------------")
        print(f"📊 Results ({args.split.upper()} split, stride={args.stride}):")
        print("  • Mean Absolute Error (MAE):       {:.2f} MW".format(mae))
        print("  • Root Mean Squared Error (RMSE):   {:.2f} MW".format(rmse))
        print("  • Mean Absolute Pct Error (MAPE):  {:.2f}%".format(mape))
        print("--------------------------------------------------------------------")
        
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
        print(f"Starting Hyperparameter Grid Search (Total configurations to check: {total_runs})")
        print(f"Sweep Grid: p={p_choices}, d={d_choices}, q={q_choices}, history={history_choices}")
        print("--------------------------------------------------------------------")
        
        leaderboard = []
        
        for idx, (order, h) in enumerate(param_grid):
            print(f"[{idx+1}/{total_runs}] Testing ARIMA{order} | History: {h} hours ... ", end="", flush=True)
            t_start = time.time()
            mae, rmse, mape = run_evaluation(
                df_unscaled, full_timestamps, indices, order, fallback_order, h, args.n_jobs
            )
            t_elapsed = time.time() - t_start
            
            if not math.isnan(mae):
                print(f"MAE: {mae:.2f} | MAPE: {mape:.2f}% | Time: {t_elapsed:.1f}s")
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
                print("FAILED/SKIPPED")
                
        # Sort leaderboard by MAE
        results_df = pd.DataFrame(leaderboard)
        if not results_df.empty:
            results_df = results_df.sort_values(by="MAE").reset_index(drop=True)
            results_df.to_csv(args.out_csv, index=False)
            
            print("\n====================================================================")
            print("🏆 ARIMA OPTIMIZATION LEADERBOARD")
            print("====================================================================")
            print(results_df[["Order", "History_Hours", "MAE", "RMSE", "MAPE", "Eval_Time_Sec"]].to_string())
            print("====================================================================")
            print(f"💾 Full results saved to: {args.out_csv}")
            print(f"🚀 Best Configuration: {results_df['Order'].iloc[0]} with {results_df['History_Hours'].iloc[0]} hours history!")
        else:
            print("⚠️ No valid configurations succeeded.")


if __name__ == "__main__":
    main()
