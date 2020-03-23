#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow.python.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, Input, Activation, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.utils import multi_gpu_model
import os
import argparse


class Caltech101(object):
    def run(self):
        tf.compat.v1.disable_eager_execution()
        # 입력 값을 받게 추가합니다.
        parser = argparse.ArgumentParser()
        parser.add_argument('--learning_rate', required=False, type=float, default=0.001)
        parser.add_argument('--dropout_rate', required=False, type=float, default=0.2)
        parser.add_argument('--batch_size', required=False, type=int, default=16)    
        parser.add_argument('--epoch', required=False, type=int, default=10)            
        # relu, sigmoid, softmax, tanh
        parser.add_argument('--act', required=False, type=str, default='relu')        
      

        args = parser.parse_args()          
        
        input = Input(shape=(200, 200, 3))
        model = InceptionV3(input_tensor=input, include_top=False, weights='imagenet', pooling='max')

        for layer in model.layers:
            layer.trainable = False

        input_image_size = (200, 200)

        x = model.output
        x = Dense(1024, name='fully')(x)
        x = Dropout(args.dropout_rate)(x)        
        x = BatchNormalization()(x)
        x = Activation(args.act)(x)
        x = Dense(512)(x)
        x = Dropout(args.dropout_rate)(x)          
        x = BatchNormalization()(x)
        x = Activation(args.act)(x)
        x = Dense(101, activation='softmax', name='softmax')(x)
        model = Model(model.input, x)

        model.summary()

        train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
        batch_size = args.batch_size

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
        model = multi_gpu_model(model, gpus=2)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=args.learning_rate),
            loss='categorical_crossentropy',
            metrics=['acc'])

        early_stopping = EarlyStopping(patience=20, mode='auto', monitor='val_acc')
        hist = model.fit_generator(train_generator,
                                      verbose=0,
                                      steps_per_epoch=train_generator.samples // batch_size,
                                      validation_data = validation_generator,
                                      epochs=args.epoch,
                                      callbacks=[early_stopping, KatibMetricLog()])
        
class KatibMetricLog(Callback):
    def on_batch_end(self, batch, logs={}):
        print("batch=" + str(batch),
              "accuracy=" + str(logs.get('acc')),
              "loss=" + str(logs.get('loss')))
    def on_epoch_begin(self, epoch, logs={}):
        print("epoch " + str(epoch) + ":")
    
    def on_epoch_end(self, epoch, logs={}):
        print("Validation-accuracy=" + str(logs.get('val_acc')),
              "Validation-loss=" + str(logs.get('val_loss')))
        return      
    
if __name__ == '__main__':
    if os.getenv('FAIRING_RUNTIME', None) is None:
        from kubeflow import fairing
        from kubeflow.fairing.kubernetes import utils as k8s_utils
        DOCKER_REGISTRY = 'kubeflow-registry.default.svc.cluster.local:30000'
        fairing.config.set_builder(
            'append',
            image_name='caltech-katib-job',
            base_image='brightfly/tf-fairing:2.0-gpu',
            registry=DOCKER_REGISTRY,
            push=True)
        # cpu 1, memory 1GiB
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
        


# In[4]:




