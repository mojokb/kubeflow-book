#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import os
import argparse
from tensorflow.python.keras.callbacks import Callback



class MyFashionMnist(object):
  def train(self):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', required=False, type=float, default=0.001)
    parser.add_argument('--dropout_rate', required=False, type=float, default=0.3)
    parser.add_argument('--opt', required=False, type=int, default=1)    
    parser.add_argument('--checkpoint_dir', required=False, default='/reuslt/training_checkpoints')
    parser.add_argument('--saved_model_dir', required=False, default='/result/saved_model')        
    parser.add_argument('--tensorboard_log', required=False, default='/result/log')     
    args = parser.parse_args()    
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.023

    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(args.dropout_rate),
      tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.summary()
    
    sgd = tf.keras.optimizers.SGD(lr=args.learning_rate)
    adam = tf.keras.optimizers.Adam(lr=args.learning_rate)
    
    optimizers= [sgd, adam]
    model.compile(optimizer=optimizers[args.opt],
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])
    
    # 체크포인트를 저장할 체크포인트 디렉터리를 지정합니다.
    checkpoint_dir = args.checkpoint_dir
    # 체크포인트 파일의 이름
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")        

    model.fit(x_train, y_train,
              verbose=0,
              validation_data=(x_test, y_test),
              epochs=5,
              callbacks=[KatibMetricLog(),
                        tf.keras.callbacks.TensorBoard(log_dir=args.tensorboard_log),
                        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                               save_weights_only=True)
                        ])
    path = args.saved_model_dir        
    model.save(path, save_format='tf')

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
            image_name='fairing-job',
            base_image='brightfly/kubeflow-jupyter-lab:tf2.0-gpu',
            registry=DOCKER_REGISTRY, 
            push=True)
        # cpu 2, memory 5GiB
        fairing.config.set_deployer('job',
                    namespace='dudaji',
                    pod_spec_mutators=[
                        k8s_utils.mounting_pvc(pvc_name="fashion-mnist", 
                                              pvc_mount_path="/result"),
                        k8s_utils.get_resource_mutator(cpu=2,
                                                       memory=5)]

                   )
        fairing.config.run()
    else:
        remote_train = MyFashionMnist()
        remote_train.train()


# In[ ]:




