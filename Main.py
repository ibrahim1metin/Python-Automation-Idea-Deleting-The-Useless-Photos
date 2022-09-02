import tensorflow as tf
import numpy as np
from io import BytesIO
from google.colab import files
import os
from PIL import Image
import tensorflow_probability as tfp
upl=files.upload()
images=[]
for _,j in upl.items():
  images.append(np.asarray(Image.open(BytesIO(j))))
for i in range(len(images)):
  images[i]=tf.image.resize(images[i],(200,300))
print(images)
def corr_sim(a,b):
  covar=tf.squeeze(tfp.stats.covariance(a,b),axis=2)
  vara=tfp.stats.variance(a)
  varb=tfp.stats.variance(b)
  return 1-tf.reduce_mean(tf.squeeze((covar/(tf.sqrt(a)*tf.sqrt(b)))))
def outlier(args):
    args=sorted(args)
    outlist=[]
    def median(args1):
        if(len(args1)%2==1):
            return list(sorted(args1))[int((len(args1)/2)-0.5)]
        else:
            return (list(sorted(args1))[int(len(args1)/2)]+list(sorted(args1))[int(len(args1)/2)-1])/2
    def minmax(data):
        if len(data) < 2: 
            raise ValueError("Data must contain at least two values to have min and max values.")
        srt = sorted(data)
        return srt[0],srt[-1]
    min,max=minmax(args)
    if len(args)%2==1:
        q1=median(args[args.index(min):args.index(median(args)):])
        q3=median(args[args.index(median(args)):args.index(max):])
        for i in args:
            if(i<(q1-1.5*(q3-q1)) or i>(q3+1.5*(q3-q1))):
                outlist.append(i)
        return outlist
    args.append(median(args))
    args=sorted(args)
    q1=median(args[args.index(min):args.index(median(args)):])
    q3=median(args[args.index(median(args)):args.index(max):])
    for i in args:
        if(i<(q1-1.5*(q3-q1)) or i>(q3+1.5*(q3-q1))):
             outlist.append(i)
    return outlist
def comparer(image_a,image_b):
  data=list()
  for i in range(image_a.shape[1]):
    for j in range(image_a.shape[-1]):
      data.append(corr_sim(image_a[::,i:i+1:,j:j+1:],image_b[::,i:i+1:,j:j+1:]))
  mean=0
  outli=outlier(data)
  for i in data:
    if not tf.math.is_inf(i) and not i in outli:
      mean+=i
  mean=mean/len(data)
  return mean
threshold=0.80
def find(arr,it):
  inx=0
  for i in arr:
    if np.all(i==it):
      break
    inx+=1
  return inx
for i in images:
  if find(images,i)==len(images)-1:
    if comparer(images[0],images[-1])<threshold:
      images.pop(images.index(i))
    else:
      continue
  else:
    print(comparer(i,images[find(images,i)+1]))
    if comparer(i,images[find(images,i)+1])<threshold:
      images.pop(find(images,i))
      print(len(images))
    else:
      continue
print(images)
