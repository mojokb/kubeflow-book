#!/usr/bin/env python
# coding: utf-8

# In[7]:


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

class SubmitKatib(object):
    def submit(self) -> str:
        
        # 입력 값을 받게 추가합니다.
        parser = argparse.ArgumentParser()
        parser.add_argument('--host', required=False, type=str, default='https://localhost:6443')                
        parser.add_argument('--token', required=False, type=str, default='token')                        
        parser.add_argument('--experiment_name', required=False, type=str, default='experiment-name')        
        args = parser.parse_args()          
        
        configuration = kubernetes.client.Configuration()
        configuration.verify_ssl=False

        configuration.host=args.host
        configuration.api_key['authorization'] = args.token

        #configuration.host='https://192.168.0.4:6443'
        #configuration.api_key['authorization'] = 'eyguswJhbGciOiJSUzI1NiIsImtpZCI6IiJ9.eyJpc3MiOiJrdWJlcm5ldGVzL3NlcnZpY2VhY2NvdW50Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9uYW1lc3BhY2UiOiJkdWRhamkiLCJrdWJlcm5ldGVzLmlvL3NlcnZpY2VhY2NvdW50L3NlY3JldC5uYW1lIjoiZGVmYXVsdC1lZGl0b3ItdG9rZW4tbGNmODciLCJrdWJlcm5ldGVzLmlvL3NlcnZpY2VhY2NvdW50L3NlcnZpY2UtYWNjb3VudC5uYW1lIjoiZGVmYXVsdC1lZGl0b3IiLCJrdWJlcm5ldGVzLmlvL3NlcnZpY2VhY2NvdW50L3NlcnZpY2UtYWNjb3VudC51aWQiOiIyMTg3NGQxYi02OGU3LTQyNTctOTE5OC1iNmYxYjViMGIwYjMiLCJzdWIiOiJzeXN0ZW06c2VydmljZWFjY291bnQ6ZHVkYWppOmRlZmF1bHQtZWRpdG9yIn0.P5FkbzyTMe8SuTh0kgM_OOOU1430A6hNjGSNEUSqIwuCnAFPfJADPK1v6DxClRZfL1emmGE72YdLY_8w7wDJPyCdcCKdO2hf7rd7G03pm_6Q6tp5xmQliMjbQQAzY_ZpyDaZFOKu5wfvXV_l5sAGAKGjIIYBr7vVObkYSHU1uU2FYs2qZlyBM_IrEFm1qLx78q57WszxT0Tg5hgoKdFig2myxmJah1z8tm61w54sWHw--8uponOFdKpDB2G_DBoWDEsmnzN9-60CY7YaEEntWDiZBv_6f6neT52S7o-xsKERB2fleynv0dCnT6oHNlJUHazfnPGqpAUKTwOyLQBZtw'
        configuration.api_key_prefix['authorization'] = 'Bearer'
        api = kubernetes.client.CustomObjectsApi(kubernetes.client.ApiClient(configuration))        

        with open("/app/katib-crd.yaml") as f:
            dep = yaml.safe_load(f)
            dep['metadata']['name'] = args.experiment_name + '-' + str(int(time.time()))

            group = 'kubeflow.org' # str | The custom resource's group name
            version = 'v1alpha3' # str | The custom resource's version
            namespace = 'dudaji' # str | The custom resource's namespace
            plural = "experiments"
            api_response = api.create_namespaced_custom_object(group=group, plural=plural, version=version, namespace=namespace, body=dep)
            print(api_response)
            with open('/tmp/result.json', 'w') as outfile:
                outfile.write(str(api_response['metadata']))
                
            with open('/result/experiment/name.txt', 'w') as outfile:
                outfile.write(str(api_response['metadata']['name']))                
            
            
if __name__ == '__main__':
    if os.getenv('FAIRING_RUNTIME', None) is None:
        from kubeflow.fairing.builders.append.append import AppendBuilder
        from kubeflow.fairing.preprocessors.converted_notebook import             ConvertNotebookPreprocessor

        DOCKER_REGISTRY = 'kubeflow-registry.default.svc.cluster.local:30000'
        base_image = 'brightfly/kubeflow-sdk-jupyter:latest'
        image_name = 'experiement-runner'
        
        katib_crd = os.path.join(os.path.dirname(os.path.realpath('__file__')), 'random-example.yaml')
        
        builder = AppendBuilder(
            registry=DOCKER_REGISTRY,
            image_name=image_name,
            base_image=base_image,
            push=True,
            preprocessor=ConvertNotebookPreprocessor(
                notebook_file="notebook_submit_experment_with_token.ipynb",
                 output_map={katib_crd: '/app/katib-crd.yaml'}
            )
        )
        builder.build()
        
    else:
        katib = SubmitKatib()
        katib.submit()


# In[ ]:




