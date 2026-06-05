#!/usr/bin/env python3
"""
Utility script to aggregate parallel SLURM Array Job ARIMA outputs
and compile a sorted hyperparameter optimization leaderboard.
"""

import os
import logging
import pandas as pd

from log_config import setup_logging

logger = logging.getLogger(__name__)

def main():
    setup_logging()
    logger.info("====================================================================")
    logger.info("Aggregating Standalone ARIMA Array Job Results")
    logger.info("====================================================================")
    
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
                logger.warning("Could not read %s: %s", filename, e)
                
    if not results:
        logger.error("No array result files (arima_res_*.csv) found.")
        logger.error("Make sure your SLURM Array Job has finished successfully.")
        return
        
    # Concatenate all results
    leaderboard = pd.concat(results, ignore_index=True)
    
    # Sort by MAE
    leaderboard = leaderboard.sort_values(by="MAE").reset_index(drop=True)
    
    # Save the consolidated results
    output_csv = "arima_opt_results.csv"
    leaderboard.to_csv(output_csv, index=False)
    
    logger.info("====================================================================")
    logger.info("CONSOLIDATED ARIMA OPTIMIZATION LEADERBOARD")
    logger.info("====================================================================")
    logger.info("\n%s", leaderboard[["Index", "Order", "History_Hours", "MAE", "RMSE", "MAPE", "Eval_Time_Sec"]].to_string())
    logger.info("====================================================================")
    logger.info("Aggregated leaderboard saved to: %s", output_csv)
    logger.info("Best ARIMA Parameters: %s with %s hours history!", leaderboard['Order'].iloc[0], leaderboard['History_Hours'].iloc[0])
    logger.info("   Metrics: MAE = %.2f | MAPE = %.2f%%", leaderboard['MAE'].iloc[0], leaderboard['MAPE'].iloc[0])
    logger.info("====================================================================")
    
    # Clean up temporary files
    logger.info("Cleaning up temporary individual csv files ...")
    for f in temp_files:
        try:
            os.remove(f)
        except Exception:
            pass
    logger.info("Cleanup complete.")


if __name__ == "__main__":
    main()
