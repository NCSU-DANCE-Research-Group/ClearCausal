import pandas as pd
import os

# Read the original Excel file
input_file = "./data/response_time_stats_history.csv"
df = pd.read_csv(input_file)
# Rename the input file to response_time_stats_history_old.csv
os.rename(input_file, "./data/response_time_stats_history_old.csv")


# Filter rows where the value in the "Name" column is "Aggregated"
result_df = df[df["Name"] == "Aggregated"]

# Save the filtered DataFrame to a new CSV file
output_file = "./data/response_time_stats_history.csv"
result_df.to_csv(output_file, index=False)

print("Filtered data saved to response_time_stats_history_new.csv")
