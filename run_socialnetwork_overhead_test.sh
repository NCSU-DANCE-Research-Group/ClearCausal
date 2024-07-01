# source /home/fqin2/.cache/pypoetry/virtualenvs/fca-init-JnQZsTKp-py3.12/bin/activate
source /home/canarypwn/.cache/pypoetry/virtualenvs/fca-init-i8TDHcjd-py3.10/bin/activate
timestamp=$(date +%s)
kubectl='minikube kubectl -p seattle --'
minikube delete -p seattle
minikube start -p seattle --cpus $(nproc)

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
# $kubectl port-forward deployment/jaeger 16686:16686 & 
$kubectl port-forward deployment/nginx-thrift 8080:8080 &
./init_user_graph.sh # may take about 26s in Seattle


cd $dir
$kubectl apply -f pod-limits.yaml
mkdir overhead_test_$timestamp
cd overhead_test_$timestamp
locust -f ~/Downloads/DeathStarBench/socialNetwork/locustfile.py --headless -u 50 --run-time 15m --spawn-rate 50 --csv=response_time --csv-full-history &
sleep 600

curl -sG 'http://127.0.0.1:9090/api/v1/query' --data-urlencode 'query=sum (rate (container_cpu_usage_seconds_total{id="/"}[5m])) / sum (machine_cpu_cores) * 100' > cpu_usage.txt

curl -sG 'http://127.0.0.1:9090/api/v1/query' --data-urlencode 'query=sum by (namespace) (rate(container_memory_working_set_bytes{namespace="default"}[5m]))' > cpu_usage.txt


