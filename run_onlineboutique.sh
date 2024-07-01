now=$(date)
./cleanup.sh
sed -i 's/^DATASET = .*/DATASET = "onlineboutique"/' setting.py
dir=$(pwd)
cd ~/Downloads/opentelemetry-online-botique-demo/
./stop_kubernetes.sh
./run_kubernetes.sh
cd $dir
mkdir data
kubectl port-forward deployment/frontend 8080:8080 &
kubectl port-forward deployment/jaeger 16686:16686 &
# background workload generator
# locust -f locustfile.py --headless -u 30 --run-time 37m --spawn-rate 40 &
sleep 60
curl "http://localhost:9090/metrics" > data/metrics-before.txt
python3 prometheus_retrieval.py
./copy_response_time_onlineboutique.sh
./get_all_logs.sh
curl "http://localhost:9090/metrics" > data/metrics-end.txt
python3 span_duration.py
python3 trace_loader.py
python3 testing_code/filter_aggregated_rows.py
# python3 calculate_correlation.py
python3 change_point.py
# python3 draw_graph.py
python3 send_email.py $now
