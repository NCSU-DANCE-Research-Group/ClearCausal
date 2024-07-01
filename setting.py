IGNORE_ERROR = True
interval_sec = 30 # seconds
# "onlineboutique" or "socialnetwork" or "mediamicroservices" or "hotelreservation"
# DATASET = "socialnetwork"
BUGID = 9
if BUGID in {2, 5, 6, 8}:
    DATASET = "socialnetwork"
elif BUGID in {1, 10}:
    DATASET = "onlineboutique"
elif BUGID in {3, 4, 7, 9}:
    DATASET = "mediamicroservices"
else:
    # dataset not set error, raise exception and exit
    raise Exception(f"Dataset for bugid {BUGID} is not set")
# True: perform root cause function analysis, False: perform root cause service analysis
ROOT_CAUSE_FUNCTION_ANALYSIS = False
