import os
import librosa
import json

a={}
b={}
basepath = r'C:\Users\Dell\Desktop\New_code\speech_data'

# for zerospeech entry[0:4]

for entry in os.listdir(basepath):
    if os.path.isfile(os.path.join(basepath, entry)):
        # dur=librosa.get_duration(filename=entry)
        if entry[0:-6] in a.keys():
            a[entry[0:-6]]+=1
            # b[entry]+=dur
        else:
            a[entry[0:-6]]=1
            # b[entry]=dur

with open("occurence.json", "w") as outfile:  
    json.dump(a, outfile) 

# with open("duration.json", "w") as outfile:  
#     json.dump(b, outfile) 
