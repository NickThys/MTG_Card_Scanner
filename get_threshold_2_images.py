from os import listdir
from statistics import mean

import cv2

results=[]
for file in listdir('debug/8_og_images_wotc'):
    og_filename='debug/8_og_images_wotc/'+file
    taken_filename='debug/9_img_taken/'+file

    og_img=cv2.resize(cv2.cvtColor(cv2.imread(og_filename),cv2.COLOR_BGR2GRAY),[600,900])
    taken_img=cv2.resize(cv2.cvtColor(cv2.imread(taken_filename),cv2.COLOR_BGR2GRAY),[600,900])

    results.append(cv2.matchTemplate(taken_img,og_img,cv2.TM_CCOEFF_NORMED))

sum=0
for res in results:
    sum+= res[0]
mean_res=sum/len(results)
print(mean_res)
