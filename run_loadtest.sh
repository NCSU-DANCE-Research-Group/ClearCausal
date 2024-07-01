trash data
mkdir data
locust -f locustfile.py --headless -u 30 --spawn-rate 40 --csv=data/response_time
