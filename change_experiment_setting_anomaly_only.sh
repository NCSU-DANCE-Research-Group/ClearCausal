# Define the file path
file="calculate_correlation.py"

# Replace the line "ANOMALY_DETECTION_MODEL = "percentile"" with "ANOMALY_DETECTION_MODEL = None"
sed -i 's/ANOMALY_DETECTION_MODEL = None/ANOMALY_DETECTION_MODEL = "percentile"/' "$file"

# Replace the line "ALIGN_ANOMALY = True" with "ALIGN_ANOMALY = False"
sed -i 's/ALIGN_ANOMALY = True/ALIGN_ANOMALY = False/' "$file"
