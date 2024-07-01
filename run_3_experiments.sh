now=$(date)
# Declare an array to store the dates
# dates=("July20" "July23" "July24") # old Bug 1:  OB email, not used
# dates=("1" "6" "23") # old Bug 1:  OB email, not used
#dates=("21" "25" "29") # new Bug 1:  OB email
# dates=("July30b" "July30c" "July30d") # old Bug 2: SocialNetwork social OOM, not used
#dates=("2" "7" "25") # new bug 2: SocialNetwork social OOM
# dates=("July31f" "Aug1d" "Aug1f") # Bug 3: Media movie OOM # "July31g" "Aug1d" "Aug1f" -> 4000, July31e, July31f -> 2000
# dates=("Aug5i" "Aug6a" "Aug6b") # Bug 4: Media userreview loop
#dates=("Apr24i" "Apr26j" "Apr27k") # Bug 5: SocialNetwork loop ("Apr3i" "Apr3j" "Apr3k")
 dates=("Aug14d" "Aug14e" "Aug14g") # Bug 6: SocialNetwork data corruption
# dates=("5" "25" "26") # Bug 7: Media data corruption, 5,12,25,26,27->1
# dates=("14" "30" "31") # Bug 8: SocialNetwork timeout, 2,5,9,14,18->1 # 14,30,31->1133 14,30,24->1122 14,31,33->1111
# dates=("4" "6" "20") # Bug 9: Media timeout, 1,3,4,6,14,17,20,23,26,27->1
# dates=("5" "7" "8") # old Bug 10: OB payment timeout, 2,17,34->0.36
# dates=("2" "17" "34") # Bug 10: OB payment timeout, 2,17,34,44->0.36; restart from 44

bugid=6  #

trash res_avg_std_

trash res_pure
trash res_wnamespace
trash res_wdependency
trash res_combined
#trash "res_pure_$date"
#trash "res_wnamespace_$date"
#trash "res_wdependency_$date"
#trash "res_combined_$date"

for date in "${dates[@]}"
do
    folder=data_B$bugid\_$date #  new B1-B2, B6-B10
#    folder="data_ob_email_$date" # old Bug 1: OB email
#    folder="data_sn_social_$date" # old Bug 2: SocialNetwork social OOM
#    folder="data_ms_movie_$date"  # Bug 3: Media movie OOM
#    folder="data_media_userreview_backfault_1min_$date" # Bug 4: Media loop
#    folder="data_bug5_$date" # Bug 5: SocialNetwork loop

    trash data
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

python3 calculate_avg_std.py "${dates[@]}"

for date in "${dates[@]}"
do
    # move the result to the corresponding folder
    trash "res_pure_$date"
    trash "res_wnamespace_$date"
    trash "res_wdependency_$date"
    trash "res_combined_$date"
done

mkdir res_avg_std_
mv *.csv res_avg_std_
# python3 send_email.py $now
