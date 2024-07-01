import os
 
# Read text File  
def read_text_file(file_path):
    first_error = None
    with open(file_path, 'r') as fin:
        for line in fin:
            # If one line contains the word "error", record the time
            if "error" in line.lower():
                if first_error is None:
                    first_error = line
    if first_error is not None:
        print(file_path, first_error)

if __name__ == "__main__":
    # Folder Path
    path = "data/logs/"
    path = os.path.join(os.getcwd(), path)
    # Change the directory
    os.chdir(path)
    # iterate through all file
    for namespace in os.listdir(path):
        print(f"Checking namespace: {namespace}")
        curr_path = os.path.join(path, namespace)
        for log_file in os.listdir(curr_path):
            # Check whether file is in text format or not
            if log_file.endswith(".log"):
                file_path = os.path.join(curr_path, log_file)
                # call read text file function
                read_text_file(file_path)
