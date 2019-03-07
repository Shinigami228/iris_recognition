#!/usr/bin/env python
# coding: utf-8

# In[14]:


from glob import glob
import os
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np

model=models.resnet18(pretrained=True)
layer=model._modules.get('avgpool')
model.eval()
scaler=transforms.Resize((224,224))
normalize=transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
to_tensor=transforms.ToTensor()
def get_vector(Image_name):
    img=Image.open(Image_name)
    t_img=Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    my_embedding=torch.zeros(1,512,1,1)
    def copy_data(m,i,o):
        my_embedding.copy_(o.data)
    h=layer.register_forward_hook(copy_data)
    model(t_img)
    h.remove()
    return my_embedding
dir_name='/home/priyanka/Desktop/biometrics project/iris/database/UBIRIS_200_150/Sessao_1/'
for filee in ['1','2','3']:
    os.chdir(dir_name+filee)
    print "hello"
    df=pd.DataFrame()
    a='/home/priyanka/Desktop/biometrics project/iris/database/UBIRIS_200_150/Sessao_1/1/*.jpg'
    for files in sorted(glob('*.jpg')):
        pic_one_vector=get_vector(files)
        pic_one_vector=np.squeeze(pic_one_vector.numpy())
        df[files]=pic_one_vector
    df.to_csv(filee+'.csv',index=False)


# In[ ]:




