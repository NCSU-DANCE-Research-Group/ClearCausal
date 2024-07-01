# Declare an array to store the dates
# dates=("July20" "July23" "July24") # B1: OB email
# dates=("July30b" "July30c" "July30d") # B2: SocialNetwork social OOM
dates=("1" "2" "3" "5" "6" "7" "13" "15" "17" "18" "19" "20" "25" "26") # new B2, 2->0.636
# dates=("July31f" "Aug1d" "Aug1f") # B3: Media movie OOM # "July31g" "Aug1d" "Aug1f" -> 4000, July31e, July31f -> 2000
# dates=("Aug6a" "Aug6b" "Aug6c") # B4: 6a and 6b: good
# dates=("Aug5i" "Aug5l" "Aug5m" "Aug5n" "Aug5o" "Aug5p") # B5
# dates=("Aug13e" "Aug13g" "Aug13i" "Aug13j" "Aug13l") # B6
# dates=("Aug14d" "Aug14e" "Aug14f" "Aug14g")
# dates=("5" "12" "25" "26" "27") # "3" "5" "11" "12" "9" "13" # B7
# dates=("2" "5" "9" "14" "18" "21" "24" "27" "29" "30" "31" "33") # "2" "5" "9" "14" "18" "21" "24" "27" "29" "30" "31" "33" # B8
# dates=("1" "3" "4" "6" "7" "8" "9" "14" "17" "20" "23" "26" "27" ) # B9
# dates=("1" "2" "3" "4" "5" "6" "7" "8") # B10
# dates=("1" "2" "6" "7" "10" "12" "14" "16" "17" "19" "20" "21" "23" "26" "27" "28") # new B1
# dates=("1" "21" "25" "29" "30") # new B1
# dates=("2" "17" "20" "24" "34") # B10
# "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23" "24"
# "25" "26" "27" "28" "29" "30" "31" "32" "33" "34" "35" "36" "37" "38" "39" "40" "41" "42" "43"


bugid=2


for date in "${dates[@]}"
do
    folder=data_B$bugid\_$date # new B1, B6-B10
    # folder="data_ob_email_$date" # old B1, OB email
    # folder="data_sn_social_$date" # B2, SocialNetwork social OOM
    # folder="data_media_userreview_backfault_1min_$date" # Media movie OOM
    # folder="data_bug5_$date" # B5

    echo $folder

    trash data
    # copy the dataset, and rename it as "data"
    cp -r "$folder" data
    # run the experiment
    python3 calculate_correlation.py False
    ./read_sheet_output.sh
    mv output.txt output_B$bugid\_$date.txt
    trash res_wnamespace && trash res_wdependency && trash res_pure
done
