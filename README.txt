#Rochelle Worsnop
#June 2, 2021

1.Download_GEFSv12_function.py
--> Function to download GEFSv12 reforcasts from AWS given inputs passed from GEFSv12_download_batch.py

2.GEFSv12_download_batch.py
run command at terminal: python -i GEFSv12_download_batch.py
--> Script that passes inputs to the GEFSv12 download script (Download_GEFSv12_function.py) for each
    date and variable.

3. Check_for_missing_downloaded_dates.py
--> Simple script to check your download directory for any missing dates. 
    It prints to the screen: missing dates and their corresponding index. 
    Use those indicies to pass through your download script again to trouble shoot errors with the data files.

#Likely will take about a week to download full GEFSv12 dataset (3 hourly, 0.25deg, 5 members) for one surface variable. 
#Tip: Run download script separately for each variable in separate terminals to get more data at once. 

