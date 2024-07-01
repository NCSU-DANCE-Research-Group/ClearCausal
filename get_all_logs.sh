for raw_namespace in $(minikube kubectl -p seattle --  get namespaces -o name); do
    namespace="${raw_namespace#*/}"
    echo $namespace
    if [ "$namespace" != 'kube-system' ] && [ "$namespace" != 'kube-public' ] && [ "$namespace" != 'kube-node-lease' ] && [ "$namespace" != 'cadvisor' ] && [ "$namespace" != 'monitoring' ] && [ "$namespace" != 'kubernetes-dashboard' ]; then
        dir=data/logs/$namespace
        mkdir -p $dir
        for raw_pod in $(minikube kubectl -p seattle --  get pods -o name --namespace=$namespace); do
            pod="${raw_pod#*/}"
            if [[ "$pod" == "${pod#otelcollector}" ]]; then
                echo $pod
                minikube kubectl -p seattle --  logs $pod --since=1h > $dir/$pod.log -n $namespace
            fi
        done
    fi
done
