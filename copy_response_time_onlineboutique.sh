pod=$(kubectl get pods -l app=loadgenerator -o jsonpath='{.items[0].metadata.name}')
for file in response_time_exceptions.csv response_time_failures.csv response_time_stats.csv response_time_stats_history.csv; do
  kubectl cp $pod:$file data/$file
done
