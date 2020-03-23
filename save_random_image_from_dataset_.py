#!/usr/bin/env python
# coding: utf-8

# In[28]:


import tensorflow as tf
import os
import numpy as np
from PIL import Image
from datetime import datetime
import random

class StoreImage(object):
  def save(self):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    
    folder_name = "./" + str(datetime.today().strftime("%Y%m%d%H%M"))
    
    # make min folder
    try:
        if not(os.path.isdir(folder_name)):
            os.makedirs(os.path.join(folder_name))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise
            
    # generate 10 ranom image (0~9999)            
    for i in range(10):
        random_num = random.randint(0, 9999)
        file_name = str(test_labels[random_num]) + "_" + str(i) + ".jpg"
        im = Image.fromarray(test_images[random_num])
        im.save(folder_name + "/" +  file_name)
    
if __name__ == '__main__':
    if os.getenv('FAIRING_RUNTIME', None) is None:
        from kubeflow import fairing
        from kubeflow.fairing.kubernetes import utils as k8s_utils

        DOCKER_REGISTRY = 'kubeflow-registry.default.svc.cluster.local:30000'
        fairing.config.set_builder(
            'append',
            image_name='store-fashion-minst',
            base_image='brightfly/kubeflow-jupyter-lab:tf2.0-cpu',
            registry=DOCKER_REGISTRY, 
            push=True)
        # cpu 2, memory 5GiB
        fairing.config.set_deployer('job',
            namespace='dudaji',
            pod_spec_mutators=[
                k8s_utils.get_resource_mutator(cpu=0.5,
                                               memory=0.5)])
         
        fairing.config.run()
    else:
        remote = StoreImage()
        remote.save()    


# In[23]:





# In[ ]:




