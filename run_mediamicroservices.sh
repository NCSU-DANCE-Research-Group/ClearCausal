now=$(date)
./cleanup.sh
sed -i 's/^DATASET = .*/DATASET = "mediamicroservices"/' setting.py
dir=$(pwd)
cd ~/Downloads/DeathStarBench/mediaMicroservices
./stop.sh
# Build the docker image
./build_docker.sh
# Load the newly built docker image to minikube
docker save -o newimage.tar yg397/media-microservices
eval $(minikube docker-env -p seattle)
docker load -i newimage.tar
# Start the application
./run.sh
sleep 60
kubectl port-forward deployment/jaeger 16686:16686 & 
kubectl port-forward deployment/nginx-web-server 8080:8080 &
./init_user_movie.sh # may take about 26s in Seattle
cd $dir
mkdir data
locust -f ~/Downloads/DeathStarBench/mediaMicroservices/locustfile.py --headless -u 50 --run-time 37m --spawn-rate 50 --csv=data/response_time --csv-full-history &
sleep 60
curl "http://localhost:9090/metrics" > data/metrics-before.txt
python3 prometheus_retrieval.py
# ./copy_response_time.sh # needs to be updated 
./get_all_logs.sh
curl "http://localhost:9090/metrics" > data/metrics-end.txt
python3 span_duration.py
python3 trace_loader.py
python3 testing_code/filter_aggregated_rows.py
python3 calculate_correlation.py
python3 change_point.py
python3 draw_graph.py
ps aux | grep "spawn-rate 40"| grep -v grep | awk {'print $2'} | xargs kill # do not kill the locust with spawn-rate 41 (sock-shop)
python3 send_email.py $now
