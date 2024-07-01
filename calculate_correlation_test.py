from calculate_correlation import *

if __name__ == "__main__":
    # Print the current experiment settings
    if DEBUG:
        print("DEBUG")
    if USE_DTW:
        print("USE_DTW")
    if FILTERING_MODE:
        print(f"FILTERING_MODE: {FILTERING_MODE}")
    if ANOMALY_DETECTION_MODEL:
        print(f"ANOMALY_DETECTION_MODEL: {ANOMALY_DETECTION_MODEL}")
    if USE_DISCRETE_MI:
        print("USE_DISCRETE_MI")
    if ROOT_CAUSE_FUNCTION_ANALYSIS:
        print("ROOT_CAUSE_FUNCTION_ANALYSIS")
    print("MINUTE_BEFORE_CHANGE_POINT: ", MINUTE_BEFORE_CHANGE_POINT)
    print("MINUTE_AFTER_CHANGE_POINT: ", MINUTE_AFTER_CHANGE_POINT)
    if USE_DISCRETE_FEATURES:
        print("DISCRETE_FEATURES")
    if ALIGN_ANOMALY:
        print("ALIGN_ANOMALY")

    if not ROOT_CAUSE_FUNCTION_ANALYSIS:
        symptom_file = "avg_span_duration_offline"
        metric_files = ["CPU_percentage_pod"]
        modes = ["wdependency"]   #
        algorithms = ["MI"]
        if DATASET == "onlineboutique":
            if BUGID == 1:
                debug_pods = [
                    "checkoutservice",
                    "emailservice-559b97c7c7-lrtg7",
                    "recommendationservice-99c856f98-z7cmw",
                    "emailservice",
                    "productcatalogservice",
                    "emailservice-696bc9f5df-42pn5",
                    "checkoutservice-55fdd74c4f-dcpjx",
                    "productcatalogservice-546b57ff5-f8gjs",
                    "cartservice",
                    "cartservice-758c5fb8cf-znqlp",
                    "emailservice-776956b7d-mnx62",
                    "checkoutservice-59f4f75568-vj7vt",
                    "productcatalogservice-56dd444df8-qjdcz",
                    "emailservice-75b579d67d-85qh6",
                    "checkoutservice-56b9c945b8-kxcfn",
                    "productcatalogservice-5b5757485d-fhb5m",
                    "currencyservice-5b67557bb6-7w9vk",
                    
                ]
            elif BUGID == 10:
                debug_pods = [
                    "checkoutservice",
                    "paymentservice",
                    # "currencyservice",
                    "paymentservice-79dffbb687-rm9lp", 
                    # "currencyservice-5bf64c85cf-ms466",
                    "alertmanager-main-1",
                    # "currencyservice",
                    # "storage-provisioner",
                    # "grafana",
                    # "alertmanager-main-2",
                    # "recommendationservice",
                    # "session-db",
                    # "shipping",
                    # "emailservice",
                    # "checkoutservice-77c787c479-ghz7h",
                    # "emailservice-67b6f6d69c-cgv69",
                    # "paymentservice-6598fc79f7-c8h2c",
                    # "currencyservice-7776f75c8f-ljppb",
                    "emailservice-58d767d8bc-95tdv"
                ]
        elif DATASET == "socialnetwork":
            if BUGID == 2:
                debug_pods = [
                    "home-timeline-service",
                    # "social-graph-service",
                    # "post-storage-service",
                    "user-mention-service",
                    # "social-graph-service-6fcddf8484-ksm2x", 
                    "social-graph-service-6fcddf8484-595k2",  # 1
                    "post-storage-service-8c74b58f4-p5sdc",  
                    "user-mention-service-cf6666994-fxwqf",
                    "social-graph-service-6fcddf8484-p6j99", # 2
                    "post-storage-service-8c74b58f4-rqw9s",
                    "user-mention-service-cf6666994-8pk5p",
                    "social-graph-service-6fcddf8484-tndcv", # 3
                    "post-storage-service-8c74b58f4-ppjhr",
                    "user-mention-service-cf6666994-62hq5"
                ]
                metric_files = ["memory"]  # ["CPU_percentage_pod"] 
            elif BUGID == 5:
                debug_pods = [
                    # "home-timeline-service",
                    # "social-graph-service-5dcf7b9796-76pn7",
                    # "home-timeline-service-56fd6b9947-k987m",
                    # "social-graph-service-67f9d7fbdb-s46mh",
                    "post-storage-service",
                    "compose-post-service",
                    # "compose-post-service-666ff97459-xkzdl",
                    # "compose-post-service-666ff97459-l72n2",
                    # "compose-post-service-666ff97459-gwh9r",
                    # "home-timeline-service",
                    # "social-graph-service-5dcf7b9796-76pn7",
                    # "home-timeline-service-56fd6b9947-k987m",
                    # "social-graph-service-67f9d7fbdb-s46mh",
                    "compose-post-service",
                    "compose-post-service-666ff97459-xkzdl",
                    "compose-post-service-666ff97459-l72n2",
                    "compose-post-service-666ff97459-gwh9r",
                    "post-storage-service",
                    "post-storage-service-5fc7969596-fnd84",
                    "post-storage-service-5fc7969596-kvst2",
                    "post-storage-service-5fc7969596-kpjzm",
                    "url-shorten-service-65cb7bb465-v6rdh",
                    # "user-mention-service",
                    # "user-mention-service-7b6987bd64-jsk8k"
                    #
                    "compose-post-service-7b6c5fc858-wqktq", # Apr13i
                    "post-storage-service-849d86667-lfxf4",  # Apr13i
                    # 
                    "post-storage-service-86547bbf4f-h7cwz", # Apr 24i...
                    "compose-post-service-7b6c5fc858-vfxsd", 
                    # "media-service-5c4df49d86-5txlp", 
                    "url-shorten-service-58d5c7c666-jkmbp", 
                    "url-shorten-mongodb-7f6fc7999b-rqnvl", 
                    "url-shorten-mongodb",
                    # "user-timeline-service-6c656f56-lgcdc", 
                    # "home-timeline-service-7888d6bcb8-9xlxc", 
                    # "social-graph-service-b64ccf865-5vt5x", 
                    # "unique-id-service-66656c57c8-5n27m", 
                    # "user-mention-service-5957dc9c5f-p6vqf", 
                    # "text-service-664f6d7c-44svr", 
                    "user-service-69479c58fc-kgzqt"
                ]
            elif BUGID == 6:
                debug_pods = [
                    "text-service",
                    "compose-post-service",
                    # "home-timeline-service",
                    # "media-service", 
                    # "unique-id-service",
                    # "user-service",
                    # "user-timeline-service",
                    "text-service-7499557c5b-f88sj",
                    "compose-post-service-666ff97459-7qpjw",
                    "home-timeline-service-56fd6b9947-vw8qt" ,
                    "social-graph-service-67f9d7fbdb-n5kvb",
                    "user-memcached-77f6976775-nmfmd",
                    "text-service-7499557c5b-6x9sr",
                    "social-graph-service-67f9d7fbdb-7dk9l",
                    "url-shorten-service-65cb7bb465-hxdlq",
                    "text-service-7499557c5b-f88sj",
                    "social-graph-service-67f9d7fbdb-n5kvb",
                    "url-shorten-memcached-7c9fb778bc-rrx2p",
                    "text-service-7499557c5b-ltf5b",  # 
                    "social-graph-service-67f9d7fbdb-b6dtr",
                    "compose-post-service-666ff97459-qlmpn"
                ]
            elif BUGID == 8:
                debug_pods = [
                    "text-service",
                    # "user-mention-service",
                    "compose-post-service-666ff97459-dhcdw",  # 1
                    "user-mention-service-7b6987bd64-52t54", 
                    "url-shorten-service-65cb7bb465-qvdkn",
                    "text-service-7499557c5b-5dq5r",
                    "grafana-589787799d-wmms4",
                    "user-mention-service-7b6987bd64-cfc55", # 
                    "compose-post-service-666ff97459-n4vh4",
                    "url-shorten-service-65cb7bb465-bkhhm",
                    "text-service-7499557c5b-nmtdb"
                ]
                # metric_files = ["CPU_percentage_pod"]  # ["memory"]
        elif DATASET == "mediamicroservices":
            if BUGID == 9:
                debug_pods = [
                    "movie-id-service/MmcGetMovieId",
                    "movie-review-service/RedisUpdate",
                    "compose-review-service/UploadText",
                    "movie-review-service/MongoFindMovie",
                    "rating-service/UploadRating",
                    "movie-id-service",
                    "rating-service",
                    "compose-review-service/UploadUserId",
                    "rating-service/RedisInsert",
                ]
                metric_files = ["operation_duration"]
                symptom_file = "avg_span_duration_offline"
                FAULTY_SERVICE_NAME = "rating-service"
    else:
        symptom_file = "CPU_percentage_pod"
        metric_files = ["operation_duration"]
        modes = ["pure"]
        algorithms = ["MI", "pearson", "spearman", "kendalltau"]
        if DATASET == "onlineboutique":
            FAULTY_SERVICE_NAME = "emailservice"
            debug_pods = [
                "emailservice",
                "emailservice-5f4954fbc7-wpk69",  # July 20
                "emailservice-7b746b4fb7-r22c2",  # July 23
                "emailservice-5c9f8cf786-5f2xq",  # July 24
                "emailservice/sendorderconfirmation",
                "emailservice-696bc9f5df-42pn5",
                "emailservice-776956b7d-mnx62",
                "emailservice-75b579d67d-85qh6"
                "emailservice/sendorderconfirmation",
                # "emailservice",
                "emailservice/jinja2.compile",
                "emailservice/grpc.health.v1.Health/Check",
                "emailservice/jinja2.load"
            ]
        elif DATASET == "socialnetwork":
            if BUGID == 2:
                debug_pods = [
                    "social-graph-service-5dcf7b9796-cdh45",  # July30b
                    "social-graph-service/get_followers_server"
                    # "home-timeline-service",
                    # # "social-graph-service",
                    # # "post-storage-service",
                    # # "social-graph-service-6fcddf8484-ksm2x", 
                    # "social-graph-service-6fcddf8484-595k2",  # 1
                    # "post-storage-service-8c74b58f4-p5sdc",  
                    # "user-mention-service-cf6666994-fxwqf",
                    # "social-graph-service-6fcddf8484-p6j99", # 2
                    # "post-storage-service-8c74b58f4-rqw9s",
                    # "social-graph-service-6fcddf8484-tndcv", # 3
                    # "post-storage-service-8c74b58f4-ppjhr"
                ]
            elif BUGID == 5:
                debug_pods = [
                    "post-storage-service-5fc7969596-fnd84",  # Aug13l
                    "post-storage-service/store_post_server"
                    # ,
                    # ""
                ]
        elif DATASET == "mediamicroservices":
            FAULTY_SERVICE_NAME = "user-review-service"
            debug_pods = [
                "user-review-service",
                "user-review-service/UploadUserReview",
            ]

    print("FAULTY_SERVICE_NAME: ", FAULTY_SERVICE_NAME)
    correlation_analysis_span_metric(symptom_file, metric_files, modes, algorithms, debug_pods, FAULTY_SERVICE_NAME,
                                     ROOT_CAUSE_NAME, ROOT_CAUSE_FUNCTION_ANALYSIS)
