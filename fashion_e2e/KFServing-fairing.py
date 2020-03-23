#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from kubernetes import client
from kfserving import KFServingClient
from kfserving import constants
from kfserving import utils
from kfserving import V1alpha2EndpointSpec
from kfserving import V1alpha2PredictorSpec
from kfserving import V1alpha2TensorflowSpec
from kfserving import V1alpha2InferenceServiceSpec
from kfserving import V1alpha2InferenceService
from kubernetes.client import V1ResourceRequirements
import os
import sys
import argparse
import logging
import time

'''
'''
class KFServing(object):
    def run(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--namespace', required=False, default='kubeflow')
        # pvc://${PVCNAME}/dir
        parser.add_argument('--storage_uri', required=False, default='/mnt/export')
        parser.add_argument('--name', required=False, default='kfserving-sample')        
        args = parser.parse_args()
        namespace = args.namespace
        serving_name =  args.name
        
        api_version = constants.KFSERVING_GROUP + '/' + constants.KFSERVING_VERSION
        default_endpoint_spec = V1alpha2EndpointSpec(
                                  predictor=V1alpha2PredictorSpec(
                                    tensorflow=V1alpha2TensorflowSpec(
                                      storage_uri=args.storage_uri,
                                      resources=V1ResourceRequirements(
                                          requests={'cpu':'100m','memory':'1Gi'},
                                          limits={'cpu':'100m', 'memory':'1Gi'}))))
        isvc = V1alpha2InferenceService(api_version=api_version,
                                  kind=constants.KFSERVING_KIND,
                                  metadata=client.V1ObjectMeta(
                                      name=serving_name, namespace=namespace),
                                  spec=V1alpha2InferenceServiceSpec(default=default_endpoint_spec))        
        
        KFServing = KFServingClient()
        KFServing.create(isvc)
        print('waiting 5 sec for Creating InferenceService')
        time.sleep(5)
        
        KFServing.get(serving_name, namespace=namespace, watch=True, timeout_seconds=300)
        
if __name__ == '__main__':
    if os.getenv('FAIRING_RUNTIME', None) is None:
        from kubeflow.fairing.builders.append.append import AppendBuilder
        from kubeflow.fairing.preprocessors.converted_notebook import             ConvertNotebookPreprocessor

        DOCKER_REGISTRY = 'kubeflow-registry.default.svc.cluster.local:30000'
        base_image = 'brightfly/kubeflow-kfserving:latest'
        image_name = 'kfserving'

        builder = AppendBuilder(
            registry=DOCKER_REGISTRY,
            image_name=image_name,
            base_image=base_image,
            push=True,
            preprocessor=ConvertNotebookPreprocessor(
                notebook_file="KFServing-fairing.ipynb"
            )
        )
        builder.build()
    else:
        serving = KFServing()
        serving.run()        
        


# In[ ]:




