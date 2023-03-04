from PIL import Image
import os
import numpy as np
import warnings
from skimage.metrics import structural_similarity
import json
warnings.simplefilter("ignore")
datas=[i for i in os.listdir(r"data")]
im_asarray=[]
files=dict()
for j in datas: 
    im=Image.open(f"data/{j}")
    im=im.resize((200,200))
    im=np.asarray(im)
    im_asarray.append(im)
def standartize(image):
    std=np.std(image)
    mean=np.mean(image) 
    image=(image-mean)/std
    return image
def find(element,array,helper=[]):
    for ind in range(len(array)):
        if  np.all(array[ind]==element) and ind not in helper: 
            return ind
    return None
def create_couples(list):
    results=[]
    for i in range(len(list)):
        for j in range(i+1,len(list)):
            results.append((list[i],list[j])) 
    return results
processed=list(map(standartize,im_asarray))
processed2=create_couples(processed)
print(create_couples(datas))
useless=[]
for couple in processed2:
    sim=structural_similarity(couple[0],couple[1],multichannel=True)
    if sim>0.6:
        useless.append(couple)
useless_to_delete=[i[1] for i in useless]
indexes=[]
for i in useless_to_delete:
    indexes.append(find(i,processed,indexes))
indexes=[i for i in indexes if i is not None]
names=[os.path.dirname(os.path.realpath(__file__))+fr"\data\{datas[i]}" for i in indexes]
as_numpy_arrays=[im_asarray[i] for i in indexes]
as_PIL=[Image.open(i) for i in names]
result={"names_and_pathes":names}
with open("jsonfile.json","w") as file:
    json.dump(result,file)
