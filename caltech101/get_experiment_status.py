#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
import time
from kubernetes import client, config
import kubernetes
from pprint import pprint
import yaml
import os
import argparse
import time
import json
import requests

class GetKatibExperimentStatus(object):
    def get(self) -> str:
        # 입력 값을 받게 추가합니다.
        parser = argparse.ArgumentParser()
        parser.add_argument('--host', required=False, type=str, default='https://localhost:6443')                
        parser.add_argument('--token', required=False, type=str, default='ey...VQ')
        parser.add_argument('--experiment_name', required=False, type=str, default='dudaji-katib-1583932201')        
        parser.add_argument('--namespace', required=False, type=str, default='dudaji')        
        args = parser.parse_args()          

        host = args.host
        headers = {'Authorization': 'Bearer ' + args.token }
        resp = requests.get(host + '/apis/kubeflow.org/v1alpha3/namespaces/' + args.namespace + '/experiments/' + args.experiment_name, 
                            headers=headers, 
                            verify=False)
        response_dict = json.loads(resp.text)
        status_dict = response_dict['status']
        condition = status_dict['conditions'][len(status_dict['conditions'])-1]['type']
        bestTrialName = status_dict['currentOptimalTrial']['bestTrialName']
        bestValidAccuracy = status_dict['currentOptimalTrial']['observation']['metrics'][0]['value']
        result = {'condition' : condition, 'bestTrialName': bestTrialName, 'bestValidAccuracy': bestValidAccuracy }
        print("result "  + str(result))
        with open('/tmp/result.json', 'w') as outfile:
            outfile.write(str(result))        
            
if __name__ == '__main__':
    if os.getenv('FAIRING_RUNTIME', None) is None:
        from kubeflow.fairing.builders.append.append import AppendBuilder
        from kubeflow.fairing.preprocessors.converted_notebook import             ConvertNotebookPreprocessor

        DOCKER_REGISTRY = 'kubeflow-registry.default.svc.cluster.local:30000'
        base_image = 'brightfly/kubeflow-sdk-jupyter:latest'
        image_name = 'get-experiment-status'
        
        builder = AppendBuilder(
            registry=DOCKER_REGISTRY,
            image_name=image_name,
            base_image=base_image,
            push=True,
            preprocessor=ConvertNotebookPreprocessor(
                notebook_file="get_experiment_status.ipynb"
            )
        )
        builder.build()
        
    else:
        status = GetKatibExperimentStatus()
        status.get()


# In[ ]:




