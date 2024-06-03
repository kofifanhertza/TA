import csv
import os
from datetime import datetime

# field names
fields = ['detection_id', 'room_id', 'timestamp', 'n_in', 'n_out']
filename = "detection_data.csv"
now = datetime.now()
date_time = now.strftime("%m/%d/%Y %H:%M:%S")


data = [['1', '1', date_time, '0','0']]


if os.path.exists(filename) :
    with open(filename, 'a', newline='') as csvfile:
    # creating a csv writer 
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(data)
else :
    with open(filename, 'w', newline='') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
    
        # writing the fields
        csvwriter.writerow(fields)
    
        # writing the data rows
        csvwriter.writerows(data)