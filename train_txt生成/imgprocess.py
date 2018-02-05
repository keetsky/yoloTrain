'''
功能:提取图片前缀名
     imgjpg        train.txt
    200004.jpg ->   200004

'''


import sys
import os

file_dir="imgjpg"     #图片目录存放在imgjpg文件夹
file_save="train.txt" #文件名保存在train.txt中
f=open(file_save,'w')


for root,dirs,files in os.walk(file_dir):
    files2=sorted(files)   #排序(如果需要打乱则不需要排序)
    print(files2)
    for file in files2:
        if os.path.splitext(file)[1]==".jpg":
            f.write(os.path.splitext(file)[0]+'\n')
            
            
f.close()
            
