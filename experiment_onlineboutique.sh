# change the dataset to be onlineboutique
./testing_code/change_dataset_onlineboutique.sh

# ./change_experiment_setting_simple.sh
# ./change_experiment_setting_anomaly_only.sh
./change_experiment_setting_anomaly_align.sh

./run_three_experiments.sh
trash data
mkdir res_avg_std
mv *.csv res_avg_std
