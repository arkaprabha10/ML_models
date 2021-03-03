import os
import json

a=open('duration_zerospeech.json')
b=json.load(a)
z={1:1,2:2}
str1="V001"
str2="V002"
c={}
for i in b.keys():
    if str1 in i:
        if str1 in c.keys():
            c[str1]+=b[i]
        else:
            c[str1]=b[i]
    elif str2 in i:
        if str2 in c.keys():
            c[str2]+=b[i]
        else:
            c[str2]=b[i]

with open("duration.json", "w") as outfile:  
    json.dump(c, outfile) 


