# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import time
import subprocess
import requests
from datetime import datetime
import pytz

def print_time():
    la_timezone = pytz.timezone('America/Los_Angeles')
    current_time_la = datetime.now(la_timezone)
    print(f"Current time: {current_time_la.strftime('%Y-%m-%d %H:%M:%S')}")

def get_jobs():
    print("=" * 60)
    print("[get_jobs] Fetching job list from squeue...")
    exec_result = subprocess.run(['squeue', '-u', os.environ.get('USER', None), '-o', '%.18i %.9P %j %.8u %.8T %.10M %.9l %.6D %R'], timeout=3600, capture_output=True, text=True)
    lines = exec_result.stdout.strip().split('\n')[1:]
    print(f"[get_jobs] Found {len(lines)} jobs in queue")
    jobs = []
    for l in lines:
        components = l.split(' ')
        components = [e for e in components if e!='']
        running_time = components[5]
        total_time = None
        time_components = running_time.split(':')
        while total_time is None:
            if '-' in time_components[0]:
                total_time = 3600
            else:
                try:
                    if len(time_components)==2:
                        total_time = int(time_components[0])*60+int(time_components[1])
                    elif len(time_components)==3:
                        total_time = int(time_components[0])*3600+int(time_components[1])*60+int(time_components[2])
                except Exception as error:
                    print(error,time_components)
                    time.sleep(10)
        jobs.append({
            'name': components[2],
            'id': components[0],
            'status': components[4],
            'total_time': total_time,
            'reason': components[-1]
        })
    print(f"[get_jobs] Parsed {len(jobs)} jobs: {[j['name'] + '(' + j['status'] + ')' for j in jobs]}")
    return jobs

EXPERIMENT_NAME1 = os.environ.get('EXPERIMENT_NAME1', 'se_t4_1')
EXPERIMENT_NAME2 = os.environ.get('EXPERIMENT_NAME2', 'se_t4_2')
EXPERIMENT_NAME3 = os.environ.get('EXPERIMENT_NAME3', 'se_t4_3')
serve_collections = [EXPERIMENT_NAME1, EXPERIMENT_NAME2, EXPERIMENT_NAME3]
print(f"[INIT] Starting resume monitor script")
print(f"[INIT] EXPERIMENT_NAME1: {EXPERIMENT_NAME1}")
print(f"[INIT] EXPERIMENT_NAME2: {EXPERIMENT_NAME2}")
print(f"[INIT] EXPERIMENT_NAME3: {EXPERIMENT_NAME3}")
print(f"[INIT] serve_collections: {serve_collections}")

loop_count = 0
while True:
    loop_count += 1
    print("\n" + "=" * 80)
    print(f"[MAIN LOOP] ========== Iteration #{loop_count} ==========")
    print_time()
    print("=" * 80)
    
    jobs = get_jobs()
    print("[HELD CHECK] Checking for held jobs...")
    for j in jobs:
        if j['reason'].strip().lower()=='held)':
            print(f"[HELD CHECK] Found held job: {j['name']} (ID: {j['id']}), cancelling...")
            os.system(f"scancel {j['id']}")
            print(f"[HELD CHECK] Cancelled job {j['id']}, sleeping 120s...")
            time.sleep(120)
    job_names = [j['name'] for j in jobs]
    print(f"[JOB NAMES] Current job names: {job_names}")
    
    print("[SERVE SUBMIT] Checking if serve jobs need to be submitted...")
    for exp_name in serve_collections:
        if not exp_name in job_names:
            print(f"[SERVE SUBMIT] {exp_name} not in job queue, submitting...")
            from filelock import FileLock
            with FileLock(f'cache/slurm_out/{exp_name}.lock'):
                if os.path.isfile(f'cache/slurm_out/{exp_name}.out'):
                    print(f"[SERVE SUBMIT] Removing old output file: cache/slurm_out/{exp_name}.out")
                    os.remove(f'cache/slurm_out/{exp_name}.out')
                print(f"[SERVE SUBMIT] Executing: sbatch {exp_name}.sh")
                os.system('sbatch '+f' {exp_name}.sh')
        else:
            print(f"[SERVE SUBMIT] {exp_name} already in queue, skipping")
    already_serve = []
    print("[SERVE STATUS] Checking running serve jobs...")
    for j in jobs:
        if j['name'] in serve_collections and j['status'].strip().lower()=='running':
            print(f"[SERVE STATUS] Found running serve job: {j['name']} (ID: {j['id']}, time: {j['total_time']}s)")
            if not os.path.isfile(f'{j["name"]}.out'):
                print(f"[SERVE STATUS] Output file {j['name']}.out not found, cancelling job {j['id']}...")
                os.system(f"scancel {j['id']}")
            else:
                if j['total_time']>=600:
                    print(f"[SERVE STATUS] {j['name']} has been running for {j['total_time']}s (>=600s), adding to already_serve")
                    already_serve.append({
                        'name': j['name'],
                        'total_time': j['total_time']
                    })
                else:
                    print(f"[SERVE STATUS] {j['name']} running time {j['total_time']}s < 600s, not ready yet")
    print(f"[SERVE STATUS] already_serve count: {len(already_serve)}/3")
    if len(already_serve)!=3:
        print(f"[SERVE STATUS] Not all 3 serve jobs ready, sleeping 30s and continuing...")
        time.sleep(30)
        continue
    all_times = [s['total_time'] for s in already_serve]
    print(f"[SERVE STATUS] All 3 serve jobs ready! Running times: {all_times}")
    if min(all_times)<600:
        wait_time = 600-min(all_times)
        print(f"[SERVE STATUS] Minimum time {min(all_times)}s < 600s, sleeping {wait_time}s...")
        time.sleep(wait_time)
    print("[IP READ] Reading serve IP addresses from output files...")
    exp_name = EXPERIMENT_NAME1
    with open(f'{exp_name}.out') as f:
        lines = f.readlines()
    serve_ip1 = lines[0].strip()
    print(f"[IP READ] {exp_name}.out -> serve_ip1: {serve_ip1}")
    
    exp_name = EXPERIMENT_NAME2
    with open(f'{exp_name}.out') as f:
        lines = f.readlines()
    serve_ip2 = lines[0].strip()
    print(f"[IP READ] {exp_name}.out -> serve_ip2: {serve_ip2}")
    
    exp_name = EXPERIMENT_NAME3
    with open(f'{exp_name}.out') as f:
        lines = f.readlines()
    serve_ip3 = lines[0].strip()
    print(f"[IP READ] {exp_name}.out -> serve_ip3: {serve_ip3}")
    print("[CONFIG CHECK] Checking if config file needs update...")
    change_flag = False
    if os.path.isfile('serve_train_tool_orchestra.json'):
        with open('serve_train_tool_orchestra.json') as f:
            old_config = json.load(f)
        print(f"[CONFIG CHECK] Old config Qwen3-32B IP: {old_config['Qwen/Qwen3-32B'][0]['ip_addr']}, new: {serve_ip2}")
        print(f"[CONFIG CHECK] Old config retrieval IP: {old_config['retrieval'][0]['ip_addr']}, new: {serve_ip1}")
        if old_config['Qwen/Qwen3-32B'][0]['ip_addr']!=serve_ip2:
            print("[CONFIG CHECK] Qwen3-32B IP changed!")
            change_flag = True
        if old_config['retrieval'][0]['ip_addr']!=serve_ip1:
            print("[CONFIG CHECK] Retrieval IP changed!")
            change_flag = True
    else:
        print("[CONFIG CHECK] Config file does not exist, need to create")
        change_flag = True
    print(f"[CONFIG CHECK] change_flag = {change_flag}")
    print("[SERVE TEST] Testing serve endpoints...")
    payload = {
        "queries": ["How to compute f(f(x)) when f is piecewise-defined"],
        "topk": 3,
        "return_scores": True,
        "eid": '84176'
    }
    serve_alive = True
    for testing_port in [1401]:
        print(f"[SERVE TEST] Testing port {testing_port} on {serve_ip1}...")
        try_count = 0
        testing_alive = False
        while not testing_alive and try_count < 5:
            try_count += 1
            print(f"[SERVE TEST] Attempt {try_count}/5 for port {testing_port}...")
            try:
                testing = requests.post(f'http://{serve_ip1}:{testing_port}/retrieve', json=payload).json()
                testing_alive = True
                print(f"[SERVE TEST] Port {testing_port} is alive! Response received.")
                break
            except Exception as serve_error:
                print(f'[SERVE TEST] Port {testing_port} serve failure on {serve_ip1}: {serve_error}')
                print(f"[SERVE TEST] Sleeping 20s before retry...")
                time.sleep(20)
        if not testing_alive:
            serve_alive = False
            print(f"[SERVE TEST] Port {testing_port} failed after 5 attempts!")
            print_time()
            jobs = get_jobs()
            job_names = [j['name'] for j in jobs]
            print(f"[SERVE TEST] Cancelling jobs starting with {EXPERIMENT_NAME1}...")
            for j in jobs:
                if j['name'].startswith(EXPERIMENT_NAME1):
                    print(f"[SERVE TEST] Cancelling job: scancel {j['id']}")
                    os.system(f"scancel {j['id']}")
            break
    if not serve_alive:
        print("[SERVE TEST] Serve not alive, restarting loop...")
        continue
    print(f'[SERVE TEST] ✓ All serve endpoints alive! serve_ip1={serve_ip1}')
    print("[CONFIG WRITE] Building model config...")
    model_config = {
        "retrieval": [{"ip_addr": serve_ip1,"port": "1401"}],
        "meta-llama/Llama-3.1-8B-Instruct": [{"ip_addr": serve_ip1,"port": "1402"}],
        "microsoft/Phi-4-mini-instruct": [{"ip_addr": serve_ip1,"port": "1408"}],
        "Qwen/Qwen2.5-Math-72B-Instruct": [{"ip_addr": serve_ip1,"port": "1403"}],
        "Qwen/Qwen2.5-Math-7B-Instruct": [{"ip_addr": serve_ip1,"port": "1404"}],
        "meta-llama/Llama-3.3-70B-Instruct": [{"ip_addr": serve_ip2,"port": "1405"}],
        "Qwen/Qwen3-32B": [{"ip_addr": serve_ip2,"port": "1406"}],
        "Qwen/Qwen2.5-Coder-32B-Instruct": [{"ip_addr": serve_ip2,"port": "1407"}],
        "google/gemma-2-9b-it": [{"ip_addr": serve_ip3,"port": "1409"}],
        "codellama/CodeLlama-7b-Instruct-hf": [{"ip_addr": serve_ip3,"port": "1410"}],
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": [{"ip_addr": serve_ip3,"port": "1411"}],
        "vllm_model_config_path": "serve_train_tool_orchestra.json"
    }
    print("[CONFIG WRITE] Writing config to serve_train_tool_orchestra.json...")
    with open('serve_train_tool_orchestra.json','w') as f:
        json.dump(model_config,f,indent=2)
    print("[CONFIG WRITE] Config written successfully!")
    print(f"[CONFIG WRITE] Model endpoints configured:")
    print(f"[CONFIG WRITE]   - serve_ip1 ({serve_ip1}): retrieval:1401, Llama-3.1-8B:1402, Phi-4-mini:1408, Qwen2.5-Math-72B:1403, Qwen2.5-Math-7B:1404")
    print(f"[CONFIG WRITE]   - serve_ip2 ({serve_ip2}): Llama-3.3-70B:1405, Qwen3-32B:1406, Qwen2.5-Coder-32B:1407")
    print(f"[CONFIG WRITE]   - serve_ip3 ({serve_ip3}): gemma-2-9b:1409, CodeLlama-7b:1410, DeepSeek-R1-Distill:1411")
    
    print(f"[TRAIN CHECK] Current job_names: {job_names}")
    if not 'train_orchestrator' in job_names:
        print('[TRAIN CHECK] train_orchestrator not in job queue, submitting train_orchestrator.sh...')
        os.system('sbatch '+'train_orchestrator.sh')
        print('[TRAIN CHECK] train_orchestrator.sh submitted!')
    else:
        print('[TRAIN CHECK] train_orchestrator already running, skipping submission')
    
    print(f"[LOOP END] Iteration #{loop_count} complete, sleeping 60s...")
    time.sleep(60)