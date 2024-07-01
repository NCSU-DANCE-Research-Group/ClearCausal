ps aux | grep deployment/jaeger | grep -v grep | awk {'print $2'} | xargs kill
ps aux | grep deployment/media-frontend | grep -v grep | awk {'print $2'} | xargs kill
ps aux | grep deployment/nginx-thrift | grep -v grep | awk {'print $2'} | xargs kill
ps aux | grep deployment/frontend | grep -v grep | awk {'print $2'} | xargs kill # online boutique
ps aux | grep deployment/nginx-web-server | grep -v grep | awk {'print $2'} | xargs kill # media service
ps aux | grep deployment.apps/jaeger-hotel-reservation-hotelres | grep -v grep | awk {'print $2'} | xargs kill # hotel reservation
ps aux | grep deployment.apps/frontend-hotel-reservation-hotelres | grep -v grep | awk {'print $2'} | xargs kill # hotel reservation
ps aux | grep "spawn-rate 50"| grep -v grep | awk {'print $2'} | xargs kill # do not kill the locust with spawn-rate 41 (sock-shop)
trash data/
trash image/
trash res/
cd ~/Downloads/opentelemetry-online-botique-demo/
./stop_kubernetes.sh
cd ~/Downloads/DeathStarBench/socialNetwork
./stop.sh
cd ~/Downloads/DeathStarBench/mediaMicroservices
./stop.sh
cd ~/Downloads/DeathStarBench/hotelReservation
./stop.sh
