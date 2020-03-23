#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.python.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, Input, Activation, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import Callback
import os

class Caltech101(object):
    def run(self):
        input = Input(shape=(200, 200, 3))
        model = InceptionV3(input_tensor=input, include_top=False, weights='imagenet', pooling='max')

        for layer in model.layers:
          layer.trainable = False

        input_image_size = (200, 200)

        x = model.output
        x = Dense(1024, name='fully')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(512)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(101, activation='softmax', name='softmax')(x)
        model = Model(model.input, x)

        model.summary()

        train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
        batch_size = 16

        train_generator = train_datagen.flow_from_directory(
            '/result/caltech101',
            target_size=input_image_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training')

        validation_generator = train_datagen.flow_from_directory(
            '/result/caltech101',
            target_size=input_image_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation')

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss='categorical_crossentropy',
            metrics=['acc'])

        early_stopping = EarlyStopping(patience=20, mode='auto', monitor='val_acc')
        hist = model.fit_generator(train_generator,
                                      steps_per_epoch=train_generator.samples // batch_size,
                                      validation_data = validation_generator,
                                      epochs=100,
                                      callbacks=[early_stopping])
        
if __name__ == '__main__':
    if os.getenv('FAIRING_RUNTIME', None) is None:
        from kubeflow import fairing
        from kubeflow.fairing.kubernetes import utils as k8s_utils
        DOCKER_REGISTRY = 'kubeflow-registry.default.svc.cluster.local:30000'
        fairing.config.set_builder(
            'append',
            image_name='caltech-fairing-job',
            base_image='brightfly/tf-fairing:2.0-gpu',
            registry=DOCKER_REGISTRY,
            push=True)
        
        fairing.config.set_deployer('job',
                                    namespace='dudaji',
                                    pod_spec_mutators=[
                                    k8s_utils.mounting_pvc(pvc_name="caltech101", 
                                                          pvc_mount_path="/result")]
                                    )
        # python3
        import IPython
        ipy = IPython.get_ipython()
        if ipy is None:
            fairing.config.set_preprocessor('python', input_files=[__file__])        
        fairing.config.run()
    else:
        train = Caltech101()
        train.run()        


# In[ ]:




