source /home/fqin2/.cache/pypoetry/virtualenvs/fca-init-JnQZsTKp-py3.12/bin/activate
kubectl='minikube kubectl -p seattle --'
minikube delete -p seattle
minikube start -p seattle --cpus $(nproc)
now=$(date)
./cleanup.sh
sed -i 's/^DATASET = .*/DATASET = "socialnetwork"/' setting.py
dir=$(pwd)

cd ~/kube-prometheus
$kubectl apply --server-side -f manifests/setup
$kubectl wait \
	--for condition=Established \
	--all CustomResourceDefinition \
	--namespace=monitoring
$kubectl apply -f manifests/
sleep 90
$kubectl --namespace monitoring port-forward svc/prometheus-k8s 9090 &

cd ~/Downloads/DeathStarBench/socialNetwork
./stop.sh
# Build the docker image
./build_docker.sh
# Load the newly built docker image to minikube
docker save -o newimage.tar deathstarbench/social-network-microservices
eval $(minikube docker-env -p seattle)
docker load -i newimage.tar
# Start the application
./run.sh
sleep 90
$kubectl port-forward deployment/media-frontend 8081:8080 & 
$kubectl port-forward deployment/jaeger 16686:16686 & 
$kubectl port-forward deployment/nginx-thrift 8080:8080 &
./init_user_graph.sh # may take about 26s in Seattle


cd $dir
$kubectl apply -f pod-limits.yaml
mkdir data
locust -f ~/Downloads/DeathStarBench/socialNetwork/locustfile.py --headless -u 50 --run-time 37m --spawn-rate 50 --csv=data/response_time --csv-full-history &
sleep 60
curl "http://localhost:9090/metrics" > data/metrics-before.txt
python3 prometheus_retrieval.py
# ./copy_response_time.sh # needs to be updated 
./get_all_logs.sh
curl "http://localhost:9090/metrics" > data/metrics-end.txt
python3 span_duration.py
python3 trace_loader.py
python3 testing_code/filter_aggregated_rows.py
# python3 calculate_correlation.py true
# python3 change_point.py
# python3 draw_graph.py
# ps aux | grep "spawn-rate 40"| grep -v grep | awk {'print $2'} | xargs kill # do not kill the locust with spawn-rate 41 (sock-shop)
# python3 send_email.py $now
