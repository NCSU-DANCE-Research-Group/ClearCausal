import pandas as pd
import os
import re
import sys
from setting import BUGID, ROOT_CAUSE_FUNCTION_ANALYSIS


# Function to extract the real service name from the column
def extract_service_name(name):
    parts = name.split("-")
    new_name = []
    for part in parts:
        # if part contains at least one digit, it is a random string, we need to stop
        if part == "cfnvt" or len(part) > 3 and re.search(r"\d", part):
            break
        else:
            new_name.append(part)
    return "-".join(new_name)

# Function to read and process a single file
def process_file(file_path):
    df = pd.read_csv(file_path)
    if not ROOT_CAUSE_FUNCTION_ANALYSIS:
        df['service_name'] = df['service_name'].apply(extract_service_name)
    # Replace empty values with 0
    df.fillna(0, inplace=True)
    return df

if __name__ == "__main__":
    # Get the command-line arguments excluding the script name
    arguments = sys.argv[1:]
    # takes a list of dates from the command line arguments
    dates = []
    if not arguments:
        print("Error: No arguments provided. Please provide a list of dates, e.g. python3 calculate_avg_std.py July30b July30c July30d")
        exit(1)
    else:
        dates = arguments
    print("dates: ", dates)
    modes = ["pure", "wnamespace", "wdependency"]
    if ROOT_CAUSE_FUNCTION_ANALYSIS:
        modes = ["pure"]
    for mode in modes:
        # List of folder names and file names
        folders = []
        for date in dates:
            folders.append(f"res_{mode}_{date}")
        file_name = "correlation_operation_duration.csv"
        if not ROOT_CAUSE_FUNCTION_ANALYSIS:
            if BUGID in {1, 4, 5, 6, 7, 8, 9, 10}:
                file_name = "correlation_CPU_percentage_pod.csv"
            elif BUGID in {2, 3}:
                file_name = "correlation_memory.csv"
            else:
                # Exit if the bug ID is not valid
                print("Error: Invalid bug ID. Please provide a valid bug ID.")
                exit(1)

        # List to store the dataframes from each file
        dfs = []

        # Process each file and store the dataframes in the dfs list
        for folder in folders:
            file_path = os.path.join(folder, file_name)
            if os.path.exists(file_path):
                df = process_file(file_path)
                dfs.append(df)

        # Combine the dataframes from all files into a single dataframe
        combined_df = pd.concat(dfs)

        # Group by service name and calculate the mean and standard deviation
        grouped_df = combined_df.groupby("service_name").agg({
            "MI": ["mean", "std"],
            "pearson": ["mean", "std"],
            "spearman": ["mean", "std"],
            "cointegration": ["mean", "std"],
            "kendalltau": ["mean", "std"]
        }).reset_index()

        # Flatten the multi-level columns
        grouped_df.columns = [' '.join(col).strip() for col in grouped_df.columns.values]

        # Save the combined dataframe to a new CSV file
        output_file = f"combined_correlation_results_{mode}.csv"
        grouped_df.to_csv(output_file, index=False)
        combined_df.to_csv(f"combined_correlation_results_all_{mode}.csv", index=False)
        print("Combined file saved as:", output_file)
