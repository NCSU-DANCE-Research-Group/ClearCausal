def parse_start_end_time():
    # read everything from the file
    with open("data/trace_start_end_time.txt", "r") as f:
        header = f.readline()
        values = f.readline()

    # Split the header and values by comma
    header_parts = header.split(", ")
    value_parts = values.split(", ")

    # Extract start time and end time values
    raw_start_time = float(value_parts[0].strip()) + 60 # start time was originally detected 1 minute, so add it back
    raw_end_time = float(value_parts[1].strip())
    return raw_start_time, raw_end_time


if __name__ == "__main__":
    start_time, end_time = parse_start_end_time()
    # Print the start time and end time
    print("Start Time:", start_time)
    print("End Time:", end_time)
