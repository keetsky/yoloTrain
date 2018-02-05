import sys
import os

file_dir="展望"
file_save="train.txt"
f=open(file_save,'w')


for root,dirs,files in os.walk(file_dir):
    files2=sorted(files)
    print(files2)
    for file in files2:
        if os.path.splitext(file)[1]==".jpg":
            f.write(os.path.splitext(file)[0]+'\n')
            
            
f.close()
            