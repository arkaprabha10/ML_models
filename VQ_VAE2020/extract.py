import os
import csv

x=set()
basepath = r'C:\Users\Dell\Desktop\New_code'
for entry in os.listdir(basepath):
    if os.path.isfile(os.path.join(basepath, entry)):
        x.add(entry[0:4])
print(len(x))
with open('speaker_list.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(x)