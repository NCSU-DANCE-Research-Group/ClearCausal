import pandas as pd
import os
from setting import IGNORE_ERROR, interval_sec, DATASET, BUGID, ROOT_CAUSE_FUNCTION_ANALYSIS

# Function to handle case 2: 12th column contains values 1, 2, and 3
def check_top_three(df, column_L_index):
    # Get the numerical index of column L (assuming it's the 12th column)
    # column_L_index = 4

    # Filter rows where column L has values of 1, 2, and 3
    filtered_df = df[df.iloc[:, column_L_index].isin([1, 2, 3])]

    # Get the values in columns A and K
    column_A_values = filtered_df.iloc[:, 0].tolist()
    column_K_values = filtered_df.iloc[:, column_L_index - 1].tolist()

    values = list(zip(column_A_values, column_K_values))

    # Sort the values based on the second column (K) in descending order
    sorted_values = sorted(values, key=lambda x: x[1], reverse=True)

    # Output the sorted values
    # print("service: correlation")
    for value_A, value_K in sorted_values:
        # We would like to remove the suffix random string of number from the service name
        parts = value_A.split("-")
        service = ""
        for part in parts:
            # if part contains a number, then it's the random string of number
            if len(part) > 1 and any(char.isdigit() for char in part):
                break
            service += part + "-"
        # remove the last dash
        service = service[:-1]
        capital_service = service[0].upper() + service[1:]
        # remove "service" from the service name to shorten it
        # capital_service = capital_service.split("service")[0].strip()

        print(f"{capital_service}: {value_K:.3f}")
    print("")

# Specify the file path
for folder in [
            # "res_combined_discrete_simple",
            # "res_combined_90_10_5",
            # "res_combined_DTW", 
            # "res_combined_DTW+85", 
            #    "res_combined_MA",
            #    "res_combined_MA+DTW", 
            #    "res_combined_mean2std", 
            #    "res_combined_nopre",
            #    "res_combined_knn_4std", 
            #    "res_combined_MA+meanstd", 
            # "res_combined_85%", 
            # "res_combined_87%",
            # "res_combined_90%",
            # "res_combined_discrete_90",
            # "res_combined_85dis", 
            # "res_combined_MA+85",
            # "res_combined_MA+85dis", 
            # "res_combined_MA+DTW+85", 
            #    "res_combined_85%", 
            #    "res_combined_meanstd",
            #    "res_combined_knn_90%", 
            #    "res_combined_2min_2min", 
            #    "res_combined_2min_1min",
            #    "res_combined_DTW+85%",
            #    "res_combined_MA+DTW+85%",
            #    "res_combined_DTW+65%",
            #    "res_combined_MA+DTW+65%",
            "res_combined",
            ]:
    print("folder: ", folder)
    file = "correlation_CPU_percentage_pod.xlsx"
    if BUGID in {2, 3}:
        file = "correlation_memory.xlsx"
    file_path = os.path.join(folder, file)

    # Read the Excel file
    df = pd.read_excel(file_path)

    print("MI")
    columns = [2, 12, 22]
    for column in columns:
        check_top_three(df, column)
    print("Pearson")
    columns = [4, 14, 24]
    for column in columns:
        check_top_three(df, column)
    print("Spearman")
    columns = [6, 16, 26]
    for column in columns:
        check_top_three(df, column)
    # print("Cointegration")
    # columns = [8, 18, 28]
    # for column in columns:
    #     check_top_three(df, column)
    print("Kendalltau")
    columns = [10, 20, 30]
    for column in columns:
        check_top_three(df, column)