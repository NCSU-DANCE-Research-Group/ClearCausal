now=$(date)

# MULTI
if [ $# -lt 2 ]; then
    echo "Usage: $0 <bug_id> <task_name>"
    exit 1
fi

bugid="$1"  # 10
task="$2"  # mi mi_ad microscope fca service function 0.2mean

dates_all=(
    "21 25 29"    # new Bug 1: OB email
    "2 7 25"      # new bug 2: SocialNetwork social OOM
    "July31f Aug1d Aug1f"  # Bug 3: Media movie OOM
    "Aug5i Aug6a Aug6b"    # Bug 4: Media userreview loop
    "Apr24i Apr26j Apr27k"  # Bug 5: SocialNetwork loop
    "Aug14d Aug14e Aug14g"  # Bug 6: SocialNetwork data corruption
    "5 25 26"      # Bug 7: Media data corruption, 5,12,25,26,27->1
    "14 30 31"     # Bug 8: SocialNetwork timeout, 2,5,9,14,18->1
    "4 6 20"       # Bug 9: Media timeout, 1,3,4,6,14,17,20,23,26,27->1
    "2 17 34"      # Bug 10: OB payment timeout, 2,17,34,44->0.36; restart from 44
)

((bug_idx = bugid - 1))
dates="${dates_all[bug_idx]}"
#echo $bugid
#echo $dates


trash res_avg_std_

trash res_pure
trash res_wnamespace
trash res_wdependency
trash res_combined
#trash "res_pure_$date"
#trash "res_wnamespace_$date"
#trash "res_wdependency_$date"
#trash "res_combined_$date"

for date in $dates # "${dates[@]}"
do
    folder=data_B$bugid\_$date #  new B1-B2, B6-B10
    if [ "$bugid" == "3" ]; then
        folder="data_ms_movie_$date"  # Bug 3: Media movie OOM
    elif [ "$bugid" == "4" ]; then
        folder="data_media_userreview_backfault_1min_$date" # Bug 4: Media loop
    elif [ "$bugid" == "5" ]; then
        folder="data_bug5_$date" # Bug 5: SocialNetwork loop
    #    else:
    #        folder=data_B$bugid\_$date #  new B1-B2, B6-B10
    #        # or
    #        # folder not set error, raise exception and exit
    #        raise Exception(f"Data folder for bugid {bugid} is not set")
    fi
#    echo $folder

    rm -rf data
    # copy the dataset, and rename it as "data"
    cp -r "$folder" data
    # update the operation data
    # python3 trace_loader.py
    # run the experiment
    python3 calculate_correlation.py True
    # move the result to the corresponding folder
    mv res_pure "res_pure_$date"
    mv res_wnamespace "res_wnamespace_$date"
    mv res_wdependency "res_wdependency_$date"
    mv res_combined "res_combined_$date"
done

python3 calculate_avg_std.py $dates # "${dates[@]}"

for date in $dates # "${dates[@]}"
do
    # move the result to the corresponding folder
    trash "res_pure_$date"
    trash "res_wnamespace_$date"
    trash "res_wdependency_$date"
    trash "res_combined_$date"
done

mkdir "res_avg_std_b${bugid}_${task}"
mv *.csv "res_avg_std_b${bugid}_${task}"
# python3 send_email.py $now

