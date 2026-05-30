#!/usr/bin/env python3
"""
Utility script to aggregate parallel SLURM Array Job ARIMA outputs
and compile a sorted hyperparameter optimization leaderboard.
"""

import os
import pandas as pd

def main():
    print("====================================================================")
    print("🏆 Aggregating Standalone ARIMA Array Job Results")
    print("====================================================================")
    
    results = []
    temp_files = []
    
    # Loop over all potential combinations (0 to 50)
    for idx in range(51):
        filename = f"arima_res_{idx}.csv"
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)
                results.append(df)
                temp_files.append(filename)
            except Exception as e:
                print(f"⚠️ Warning: Could not read {filename}: {e}")
                
    if not results:
        print("❌ No array result files (arima_res_*.csv) found.")
        print("Make sure your SLURM Array Job has finished successfully.")
        return
        
    # Concatenate all results
    leaderboard = pd.concat(results, ignore_index=True)
    
    # Sort by MAE
    leaderboard = leaderboard.sort_values(by="MAE").reset_index(drop=True)
    
    # Save the consolidated results
    output_csv = "arima_opt_results.csv"
    leaderboard.to_csv(output_csv, index=False)
    
    print("\n====================================================================")
    print("🥇 CONSOLIDATED ARIMA OPTIMIZATION LEADERBOARD")
    print("====================================================================")
    print(leaderboard[["Index", "Order", "History_Hours", "MAE", "RMSE", "MAPE", "Eval_Time_Sec"]].to_string())
    print("====================================================================")
    print(f"💾 Aggregated leaderboard saved to: {output_csv}")
    print(f"🚀 Best ARIMA Parameters: {leaderboard['Order'].iloc[0]} with {leaderboard['History_Hours'].iloc[0]} hours history!")
    print(f"   Metrics: MAE = {leaderboard['MAE'].iloc[0]:.2f} | MAPE = {leaderboard['MAPE'].iloc[0]:.2f}%")
    print("====================================================================")
    
    # Clean up temporary files
    print("\nCleaning up temporary individual csv files ...")
    for f in temp_files:
        try:
            os.remove(f)
        except Exception:
            pass
    print("🧹 Cleanup complete.")


if __name__ == "__main__":
    main()
