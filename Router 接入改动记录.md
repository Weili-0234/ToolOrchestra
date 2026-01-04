# Router 接入改动记录

**总目的**：让ToolOrchestra评测请求走 `multinode_thunder.py`，并让 router 能正确识别同一 program、进行粘性路由与资源释放。

## 1) vLLM 请求里增加 job_id / is_last_step
**文件**：`LLM_CALL.py`  
**改动**：在 vLLM 分支调用里把 job_id / is_last_step 写入 `extra_body`。  
**目的**：让 router 能读取并做粘性路由与清理。

---

## 2) HLE / FRAMES 评测传递 job_id 与 is_last_step
**文件**：`evaluation/eval_hle.py`、`evaluation/eval_hle_basic.py`、`evaluation/eval_frames.py`  
**改动**：
- 在 `run_single` 里生成 job_id：  
  - HLE：`hle:{e['id']}`  
  - FRAMES：`frames:{e['id']}`
- 在构造 `call_tool_argument` 时附带 job_id / is_last_step。  
  - is_last_step 判定：`tool == "answer"` 或 `step == MAX_ROUNDS - 1`  
  **目的**：让 router 能稳定识别每条样本的 program 并在结束时清理。

---

## 3) tau2-bench 评测传递 job_id，并用 release 精准释放
**文件**：`evaluation/tau2-bench/tau2/orchestrator/orchestrator.py`、`evaluation/tau2-bench/tau2/utils/llm_utils.py`  
**改动**：
- 在 Orchestrator 初始化时生成 job_id：  
  `tau2:{domain}:{task.id}:{uuid}`  
- 每次 LLM 调用前把 job_id 写入 agent/user 的 `llm_args`，由 `llm_utils.py` 透传到 vLLM 请求。  
- 因为结束只能在回复后确定、调用前的猜测不可靠，所以 tau2-bench 不再传 is_last_step，释放统一走 release：  
  - 收到 STOP 立刻 POST `/programs/release`  
  - run() finally 兜底调用 `/programs/release`
  **目的**：STOP 只有在回复后才知道，显式 release 更可靠。

---

## 4) router 增加 release 接口
**文件**：`multinode_thunder.py`  
**改动**：新增 `POST /programs/release`，传 `job_id` 即可清理状态。  
**目的**：给 tau2-bench 做显式释放。

---

---

## 5) Patch
**LLM_CALL.py**
````diff
diff --git a/LLM_CALL.py b/LLM_CALL.py
index 7ae24ef..93d63a5 100644
--- a/LLM_CALL.py
+++ b/LLM_CALL.py
@@ -288,7 +288,18 @@ def get_openai_client(model):
     )
     return client
 
-def get_llm_response(model,messages,temperature=1.0,return_raw_response=False,tools=None,show_messages=False,model_type=None,max_length=1024,model_config=None,model_config_idx=0,model_config_path=None,payload=None,**kwargs):
+def _build_router_extra_body(extra_body, job_id, is_last_step):
+    body = {}
+    if isinstance(extra_body, dict):
+        body.update(extra_body)
+    if job_id is not None:
+        body["job_id"] = str(job_id)
+    if is_last_step is not None:
+        body["is_last_step"] = bool(is_last_step)
+    return body or None
+
+
+def get_llm_response(model,messages,temperature=1.0,return_raw_response=False,tools=None,show_messages=False,model_type=None,max_length=1024,model_config=None,model_config_idx=0,model_config_path=None,payload=None,job_id=None,is_last_step=None,extra_body=None,**kwargs):
     if isinstance(messages,str):
         messages = [{'role': 'user','content': messages}]
     if model in ['o3','o3-mini','gpt-4o','o3-high','gpt-5','gpt-5-mini','gpt-4.1','gpt-4o-mini']:
@@ -363,13 +374,17 @@ def get_llm_response(model,messages,temperature=1.0,return_raw_response=False,to
                     api_key="EMPTY",
                     base_url=f"http://{ip_addr}:{port}/v1",
                 )
-                chat_completion = vllm_client.chat.completions.create(
-                    model=model,
-                    messages=messages,
-                    max_tokens=max_length,
-                    temperature=temperature,
-                    tools=tools
-                )
+                request_kwargs = {
+                    "model": model,
+                    "messages": messages,
+                    "max_tokens": max_length,
+                    "temperature": temperature,
+                    "tools": tools,
+                }
+                router_extra_body = _build_router_extra_body(extra_body, job_id, is_last_step)
+                if router_extra_body:
+                    request_kwargs["extra_body"] = router_extra_body
+                chat_completion = vllm_client.chat.completions.create(**request_kwargs)
                 if return_raw_response:
                     answer = chat_completion
                 else:
@@ -437,4 +452,3 @@ def get_llm_response(model,messages,temperature=1.0,return_raw_response=False,to
                 time.sleep(60)
         return answer
 
-
````

**evaluation/eval_frames.py**
````diff
diff --git a/evaluation/eval_frames.py b/evaluation/eval_frames.py
index a5f106b..a9ece59 100644
--- a/evaluation/eval_frames.py
+++ b/evaluation/eval_frames.py
@@ -169,6 +169,8 @@ def cut_seq(seq,l):
 
 def call_tool(arguments):
     start_time = time.time()
+    job_id = arguments.get("job_id")
+    is_last_step = bool(arguments.get("is_last_step", False))
     if arguments['tool']=='enhance_reasoning':
         supported_models = [MODEL_MAPPING[m] for m in ALL_TOOLS['enhance_reasoning']['model']]
         assert arguments['model'] in supported_models,f"Model {arguments['model']} is not supported in enhance_reasoning. Support models: {supported_models}"
@@ -179,7 +181,7 @@ def call_tool(arguments):
         if 'gpt-5' in model_name.lower() or 'claude' in model_name.lower():
             response = get_llm_response(model=model_name,messages=prompt,return_raw_response=True,temperature=1,max_length=40000)
         elif 'qwen2.5-coder' in model_name.lower() or 'nemotron' in model_name.lower() or '235' in model_name.lower():
-            response = get_llm_response(model=model_name,messages=prompt,return_raw_response=True,model_type='vllm',max_length=8000,temperature=0.2,model_config=arguments['vllm_model_configs'][model_name],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['eid'])
+            response = get_llm_response(model=model_name,messages=prompt,return_raw_response=True,model_type='vllm',max_length=8000,temperature=0.2,model_config=arguments['vllm_model_configs'][model_name],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['eid'],job_id=job_id,is_last_step=is_last_step)
             if isinstance(response,str):
                 response = ''
                 while not response:
@@ -193,7 +195,7 @@ def call_tool(arguments):
                     except Exception as qwen_error:
                         time.sleep(3)
         elif 'qwen3-8b' in model_name.lower() or 'llama-3.3' in model_name.lower():
-            response = get_llm_response(model=model_name,messages=prompt,return_raw_response=True,model_type='vllm',max_length=8000,temperature=0.2,model_config=arguments['vllm_model_configs'][model_name],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['eid'])
+            response = get_llm_response(model=model_name,messages=prompt,return_raw_response=True,model_type='vllm',max_length=8000,temperature=0.2,model_config=arguments['vllm_model_configs'][model_name],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['eid'],job_id=job_id,is_last_step=is_last_step)
         if isinstance(response,str):
             arguments['generated_code'] = ''
             arguments['exec_result'] = ''
@@ -239,7 +241,7 @@ def call_tool(arguments):
                 {"role": "user", "content": prompt}
             ]
             arguments['messages'] = messages
-            response = get_llm_response(model=model_name,messages=messages,return_raw_response=True,model_type='vllm',max_length=8000,temperature=0.2,model_config=arguments['vllm_model_configs'][model_name],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['eid'])
+            response = get_llm_response(model=model_name,messages=messages,return_raw_response=True,model_type='vllm',max_length=8000,temperature=0.2,model_config=arguments['vllm_model_configs'][model_name],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['eid'],job_id=job_id,is_last_step=is_last_step)
             if isinstance(response,str):
                 arguments['response'] = ''
                 arguments['pred'] = ''
@@ -258,7 +260,7 @@ def call_tool(arguments):
                 {"role": "user", "content": prompt}
             ]
             arguments['messages'] = messages
-            response = get_llm_response(model=model_name,messages=messages,return_raw_response=True,model_type='vllm',max_length=2000,temperature=0.2,model_config=arguments['vllm_model_configs'][model_name],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['eid'])
+            response = get_llm_response(model=model_name,messages=messages,return_raw_response=True,model_type='vllm',max_length=2000,temperature=0.2,model_config=arguments['vllm_model_configs'][model_name],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['eid'],job_id=job_id,is_last_step=is_last_step)
             if isinstance(response,str):
                 arguments['response'] = ''
                 arguments['pred'] = ''
@@ -289,7 +291,7 @@ def call_tool(arguments):
             model_name = arguments['model']
             prompt += "\nWrap the thinking process and explanation between <think> and </think> and wrap only the exact answer without any explanation within <answer> and </answer>."
             arguments['messages'] = prompt
-            response = get_llm_response(model=model_name,messages=prompt,return_raw_response=True,model_type='vllm',max_length=40000,temperature=0.2,model_config=arguments['vllm_model_configs'][model_name],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['eid'])
+            response = get_llm_response(model=model_name,messages=prompt,return_raw_response=True,model_type='vllm',max_length=40000,temperature=0.2,model_config=arguments['vllm_model_configs'][model_name],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['eid'],job_id=job_id,is_last_step=is_last_step)
             if isinstance(response,str):
                 response = ''
                 while not response:
@@ -352,7 +354,7 @@ def call_tool(arguments):
             else:
                 query_to_call = response['content'][0]['text'].split('<query>')[-1].split('</query>')[0]
         elif 'qwen3' in cur_query_writer.lower():
-            response = get_llm_response(model=cur_query_writer,messages=prompt,return_raw_response=True,model_type='vllm',max_length=8000,temperature=0.2,model_config=arguments['vllm_model_configs'][cur_query_writer],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['eid'])
+            response = get_llm_response(model=cur_query_writer,messages=prompt,return_raw_response=True,model_type='vllm',max_length=8000,temperature=0.2,model_config=arguments['vllm_model_configs'][cur_query_writer],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['eid'],job_id=job_id,is_last_step=is_last_step)
             if isinstance(response,str):
                 query_to_call = arguments['problem']
             else:
@@ -460,6 +462,7 @@ def run_single(e):
     problem = e['question']
     user_problem = problem
     answer = e['answer']
+    job_id = f"frames:{e['id']}"
     all_tool_calls = []
     final_correct = False
     final_answer_model = None
@@ -522,7 +525,7 @@ def run_single(e):
                     {"role": "system", "content": "You are good at using tools."},
                     {"role": "user", "content": f"Problem: {problem}\n\n{context_str}\n\nChoose an appropriate tool."}
                 ]
-        response = get_llm_response(model=MODEL_NAME,messages=chat,return_raw_response=True,model_type='vllm',model_config=vllm_model_configs[MODEL_NAME],temperature=1,max_length=12000,tools=tools,model_config_path=vllm_model_configs['vllm_model_config_path'],model_config_idx=e['eid'])
+        response = get_llm_response(model=MODEL_NAME,messages=chat,return_raw_response=True,model_type='vllm',model_config=vllm_model_configs[MODEL_NAME],temperature=1,max_length=12000,tools=tools,model_config_path=vllm_model_configs['vllm_model_config_path'],model_config_idx=e['eid'],job_id=job_id,is_last_step=False)
         
         if isinstance(response,str):
             continue
@@ -597,7 +600,9 @@ def run_single(e):
                     'cur_output_dir': cur_output_dir,
                     'problem': user_problem,
                     'id': e['id'],
-                    'eid': e['eid']
+                    'eid': e['eid'],
+                    'job_id': job_id,
+                    'is_last_step': tool_name == 'answer' or step == MAX_ROUNDS - 1
                 }
             elif tool_call['name']=='answer':
                 if 'qwen2.5-math' in expert_model_to_call.lower():
@@ -637,7 +642,9 @@ def run_single(e):
                     'problem': user_problem,
                     'answer': answer,
                     'id': e['id'],
-                    'eid': e['eid']
+                    'eid': e['eid'],
+                    'job_id': job_id,
+                    'is_last_step': tool_name == 'answer' or step == MAX_ROUNDS - 1
                 }
             elif tool_call['name'] in ['search']:
                 if 'qwen3' in expert_model_to_call.lower():
@@ -671,7 +678,9 @@ def run_single(e):
                     'problem': user_problem,
                     'answer': answer,
                     'id': e['id'],
-                    'eid': e['eid']
+                    'eid': e['eid'],
+                    'job_id': job_id,
+                    'is_last_step': tool_name == 'answer' or step == MAX_ROUNDS - 1
                 }
             tool_call_list.append([call_tool,call_tool_argument])
             if tool_call['name']=='answer':
````

**evaluation/eval_hle.py**
````diff
diff --git a/evaluation/eval_hle.py b/evaluation/eval_hle.py
index deeaec0..d16e243 100644
--- a/evaluation/eval_hle.py
+++ b/evaluation/eval_hle.py
@@ -128,6 +128,8 @@ def cut_seq(seq,l):
 
 def call_tool(arguments):
     start_time = time.time()
+    job_id = arguments.get("job_id")
+    is_last_step = bool(arguments.get("is_last_step", False))
     if arguments['tool']=='enhance_reasoning':
         supported_models = [MODEL_MAPPING[m] for m in ALL_TOOLS['enhance_reasoning']['model']]
         assert arguments['model'] in supported_models,f"Model {arguments['model']} is not supported in enhance_reasoning. Support models: {supported_models}"
@@ -138,7 +140,7 @@ def call_tool(arguments):
         if 'gpt-5' in model_name.lower():
             response = get_llm_response(model=model_name,messages=prompt,return_raw_response=True,temperature=1,max_length=40000)
         elif 'qwen2.5-coder' in model_name.lower():
-            response = get_llm_response(model=model_name,messages=prompt,return_raw_response=True,model_type='vllm',max_length=8000,temperature=0.2,model_config=arguments['vllm_model_configs'][model_name],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['eid'])
+            response = get_llm_response(model=model_name,messages=prompt,return_raw_response=True,model_type='vllm',max_length=8000,temperature=0.2,model_config=arguments['vllm_model_configs'][model_name],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['eid'],job_id=job_id,is_last_step=is_last_step)
             if isinstance(response,str):
                 response = ''
                 while not response:
@@ -193,7 +195,7 @@ def call_tool(arguments):
                 {"role": "user", "content": prompt}
             ]
             arguments['messages'] = messages
-            response = get_llm_response(model=model_name,messages=messages,return_raw_response=True,model_type='vllm',max_length=8000,temperature=0.2,model_config=arguments['vllm_model_configs'][model_name],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['eid'])
+            response = get_llm_response(model=model_name,messages=messages,return_raw_response=True,model_type='vllm',max_length=8000,temperature=0.2,model_config=arguments['vllm_model_configs'][model_name],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['eid'],job_id=job_id,is_last_step=is_last_step)
             if isinstance(response,str):
                 arguments['response'] = ''
                 arguments['pred'] = ''
@@ -212,7 +214,7 @@ def call_tool(arguments):
                 {"role": "user", "content": prompt}
             ]
             arguments['messages'] = messages
-            response = get_llm_response(model=model_name,messages=messages,return_raw_response=True,model_type='vllm',max_length=2000,temperature=0.2,model_config=arguments['vllm_model_configs'][model_name],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['eid'])
+            response = get_llm_response(model=model_name,messages=messages,return_raw_response=True,model_type='vllm',max_length=2000,temperature=0.2,model_config=arguments['vllm_model_configs'][model_name],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['eid'],job_id=job_id,is_last_step=is_last_step)
             if isinstance(response,str):
                 arguments['response'] = ''
                 arguments['pred'] = ''
@@ -244,7 +246,7 @@ def call_tool(arguments):
             model_name = arguments['model']
             prompt += "\nWrap the thinking process and explanation between <think> and </think> and wrap only the exact answer without any explanation within <answer> and </answer>."
             arguments['messages'] = prompt
-            response = get_llm_response(model=model_name,messages=prompt,return_raw_response=True,model_type='vllm',max_length=40000,temperature=0.2,model_config=arguments['vllm_model_configs'][model_name],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['eid'])
+            response = get_llm_response(model=model_name,messages=prompt,return_raw_response=True,model_type='vllm',max_length=40000,temperature=0.2,model_config=arguments['vllm_model_configs'][model_name],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['eid'],job_id=job_id,is_last_step=is_last_step)
             if isinstance(response,str):
                 response = ''
                 while not response:
@@ -301,7 +303,7 @@ def call_tool(arguments):
             else:
                 query_to_call = response.choices[0].message.content.split('<query>')[-1].split('</query>')[0]
         elif 'qwen3' in cur_query_writer.lower():
-            response = get_llm_response(model=cur_query_writer,messages=prompt,return_raw_response=True,model_type='vllm',max_length=8000,temperature=0.2,model_config=arguments['vllm_model_configs'][cur_query_writer],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['eid'])
+            response = get_llm_response(model=cur_query_writer,messages=prompt,return_raw_response=True,model_type='vllm',max_length=8000,temperature=0.2,model_config=arguments['vllm_model_configs'][cur_query_writer],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['eid'],job_id=job_id,is_last_step=is_last_step)
             if isinstance(response,str):
                 query_to_call = arguments['problem']
             else:
@@ -410,6 +412,7 @@ def run_single(e):
     problem = e['question']
     user_problem = problem
     answer = e['answer']
+    job_id = f"hle:{e['id']}"
     all_tool_calls = []
     final_correct = False
     final_answer_model = None
@@ -473,7 +476,7 @@ def run_single(e):
                     {"role": "system", "content": "You are good at using tools."},
                     {"role": "user", "content": f"Problem: {problem}\n\n{context_str}\n\nChoose an appropriate tool.'"}
                 ]
-        response = get_llm_response(model=MODEL_NAME,messages=chat,return_raw_response=True,model_type='vllm',model_config=vllm_model_configs[MODEL_NAME],temperature=1,max_length=12000,tools=tools,model_config_path=vllm_model_configs['vllm_model_config_path'],model_config_idx=e['eid'])
+        response = get_llm_response(model=MODEL_NAME,messages=chat,return_raw_response=True,model_type='vllm',model_config=vllm_model_configs[MODEL_NAME],temperature=1,max_length=12000,tools=tools,model_config_path=vllm_model_configs['vllm_model_config_path'],model_config_idx=e['eid'],job_id=job_id,is_last_step=False)
         cache_idx = 0
         while os.path.isfile(f"input_output/{cache_idx}.json"):
             cache_idx += 1
@@ -567,7 +570,9 @@ def run_single(e):
                     'cur_output_dir': cur_output_dir,
                     'problem': user_problem,
                     'id': e['id'],
-                    'eid': e['eid']
+                    'eid': e['eid'],
+                    'job_id': job_id,
+                    'is_last_step': tool_name == 'answer' or step == MAX_ROUNDS - 1
                 }
             elif tool_call['name']=='answer':
                 if 'qwen2.5-math' in expert_model_to_call.lower():
@@ -610,7 +615,9 @@ def run_single(e):
                     'problem': user_problem,
                     'answer': answer,
                     'id': e['id'],
-                    'eid': e['eid']
+                    'eid': e['eid'],
+                    'job_id': job_id,
+                    'is_last_step': tool_name == 'answer' or step == MAX_ROUNDS - 1
                 }
             elif tool_call['name'] in ['search']:
                 if 'qwen3' in expert_model_to_call.lower():
@@ -647,7 +654,9 @@ def run_single(e):
                     'problem': user_problem,
                     'answer': answer,
                     'id': e['id'],
-                    'eid': e['eid']
+                    'eid': e['eid'],
+                    'job_id': job_id,
+                    'is_last_step': tool_name == 'answer' or step == MAX_ROUNDS - 1
                 }
             tool_call_list.append([call_tool,call_tool_argument])
             break
````

**evaluation/eval_hle_basic.py**
````diff
diff --git a/evaluation/eval_hle_basic.py b/evaluation/eval_hle_basic.py
index 5e98736..358d0b9 100644
--- a/evaluation/eval_hle_basic.py
+++ b/evaluation/eval_hle_basic.py
@@ -178,6 +178,8 @@ ALL_TOOLS = {
 
 def call_tool(arguments):
     start_time = time.time()
+    job_id = arguments.get("job_id")
+    is_last_step = bool(arguments.get("is_last_step", False))
     if arguments['tool']=='enhance_reasoning':
         prompt = arguments['context_str'].strip()+'\n\n'
         prompt += f"Question: {arguments['problem']}\nInstead of directly answering the question, please write additional python code that will give intermidiate results after execution. Wrap the code within ```python and ```. The code should be self-contained with all the import and initialization."
@@ -186,7 +188,7 @@ def call_tool(arguments):
         if 'gpt-5' in model_name.lower():
             response = get_llm_response(model=model_name,messages=prompt,return_raw_response=True,temperature=1,max_length=40000)
         elif 'qwen2.5-coder' in model_name.lower() or model_name == MODEL_NAME:
-            response = get_llm_response(model=model_name,messages=prompt,return_raw_response=True,model_type='vllm',max_length=8000,temperature=0.2,model_config=arguments['vllm_model_configs'][model_name],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['eid'])
+            response = get_llm_response(model=model_name,messages=prompt,return_raw_response=True,model_type='vllm',max_length=8000,temperature=0.2,model_config=arguments['vllm_model_configs'][model_name],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['eid'],job_id=job_id,is_last_step=is_last_step)
             if isinstance(response,str):
                 response = ''
                 while not response:
@@ -241,7 +243,7 @@ def call_tool(arguments):
                 {"role": "user", "content": prompt}
             ]
             arguments['messages'] = messages
-            response = get_llm_response(model=model_name,messages=messages,return_raw_response=True,model_type='vllm',max_length=8000,temperature=0.2,model_config=arguments['vllm_model_configs'][model_name],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['eid'])
+            response = get_llm_response(model=model_name,messages=messages,return_raw_response=True,model_type='vllm',max_length=8000,temperature=0.2,model_config=arguments['vllm_model_configs'][model_name],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['eid'],job_id=job_id,is_last_step=is_last_step)
             if isinstance(response,str):
                 arguments['response'] = ''
                 arguments['pred'] = ''
@@ -260,7 +262,7 @@ def call_tool(arguments):
                 {"role": "user", "content": prompt}
             ]
             arguments['messages'] = messages
-            response = get_llm_response(model=model_name,messages=messages,return_raw_response=True,model_type='vllm',max_length=2000,temperature=0.2,model_config=arguments['vllm_model_configs'][model_name],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['eid'])
+            response = get_llm_response(model=model_name,messages=messages,return_raw_response=True,model_type='vllm',max_length=2000,temperature=0.2,model_config=arguments['vllm_model_configs'][model_name],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['eid'],job_id=job_id,is_last_step=is_last_step)
             if isinstance(response,str):
                 arguments['response'] = ''
                 arguments['pred'] = ''
@@ -292,7 +294,7 @@ def call_tool(arguments):
             model_name = arguments['model']
             prompt += "\nWrap the thinking process and explanation between <think> and </think> and wrap only the exact answer without any explanation within <answer> and </answer>."
             arguments['messages'] = prompt
-            response = get_llm_response(model=model_name,messages=prompt,return_raw_response=True,model_type='vllm',max_length=40000,temperature=0.2,model_config=arguments['vllm_model_configs'][model_name],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['eid'])
+            response = get_llm_response(model=model_name,messages=prompt,return_raw_response=True,model_type='vllm',max_length=40000,temperature=0.2,model_config=arguments['vllm_model_configs'][model_name],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['eid'],job_id=job_id,is_last_step=is_last_step)
             if isinstance(response,str):
                 response = ''
                 while not response:
@@ -349,7 +351,7 @@ def call_tool(arguments):
             else:
                 query_to_call = response.choices[0].message.content.split('<query>')[-1].split('</query>')[0]
         elif 'qwen3' in cur_query_writer.lower() or model_name==MODEL_NAME:
-            response = get_llm_response(model=cur_query_writer,messages=prompt,return_raw_response=True,model_type='vllm',max_length=8000,temperature=0.2,model_config=arguments['vllm_model_configs'][cur_query_writer],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['eid'])
+            response = get_llm_response(model=cur_query_writer,messages=prompt,return_raw_response=True,model_type='vllm',max_length=8000,temperature=0.2,model_config=arguments['vllm_model_configs'][cur_query_writer],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['eid'],job_id=job_id,is_last_step=is_last_step)
             if isinstance(response,str):
                 query_to_call = arguments['problem']
             else:
@@ -408,6 +410,7 @@ def run_single(e):
     problem = e['question']
     user_problem = problem
     answer = e['answer']
+    job_id = f"hle:{e['id']}"
     all_tool_calls = []
     final_correct = False
     final_answer_model = None
@@ -470,7 +473,7 @@ def run_single(e):
                     {"role": "system", "content": "You are good at using tools."},
                     {"role": "user", "content": f"Problem: {problem}\n\n{context_str}\n\nChoose an appropriate tool.'"}
                 ]
-        response = get_llm_response(model=MODEL_NAME,messages=chat,return_raw_response=True,model_type='vllm',model_config=vllm_model_configs[MODEL_NAME],temperature=1,max_length=12000,tools=tools,model_config_path=vllm_model_configs['vllm_model_config_path'],model_config_idx=e['eid'])
+        response = get_llm_response(model=MODEL_NAME,messages=chat,return_raw_response=True,model_type='vllm',model_config=vllm_model_configs[MODEL_NAME],temperature=1,max_length=12000,tools=tools,model_config_path=vllm_model_configs['vllm_model_config_path'],model_config_idx=e['eid'],job_id=job_id,is_last_step=False)
         cache_idx = 0
         while os.path.isfile(f"input_output/{cache_idx}.json"):
             cache_idx += 1
@@ -549,7 +552,9 @@ def run_single(e):
                     'cur_output_dir': cur_output_dir,
                     'problem': user_problem,
                     'id': e['id'],
-                    'eid': e['eid']
+                    'eid': e['eid'],
+                    'job_id': job_id,
+                    'is_last_step': tool_name == 'answer' or step == MAX_ROUNDS - 1
                 }
             elif tool_call['name']=='answer':
                 if 'qwen2.5-math' in expert_model_to_call.lower() or expert_model_to_call == MODEL_NAME:
@@ -592,7 +597,9 @@ def run_single(e):
                     'problem': user_problem,
                     'answer': answer,
                     'id': e['id'],
-                    'eid': e['eid']
+                    'eid': e['eid'],
+                    'job_id': job_id,
+                    'is_last_step': tool_name == 'answer' or step == MAX_ROUNDS - 1
                 }
             elif tool_call['name'] in ['search']:
                 if 'qwen3' in expert_model_to_call.lower() or expert_model_to_call == MODEL_NAME:
@@ -629,7 +636,9 @@ def run_single(e):
                     'problem': user_problem,
                     'answer': answer,
                     'id': e['id'],
-                    'eid': e['eid']
+                    'eid': e['eid'],
+                    'job_id': job_id,
+                    'is_last_step': tool_name == 'answer' or step == MAX_ROUNDS - 1
                 }
             tool_call_list.append([call_tool,call_tool_argument])
             break
````

**evaluation/tau2-bench/tau2/orchestrator/orchestrator.py**
````diff
diff --git a/evaluation/tau2-bench/tau2/orchestrator/orchestrator.py b/evaluation/tau2-bench/tau2/orchestrator/orchestrator.py
index 49dba08..a469388 100644
--- a/evaluation/tau2-bench/tau2/orchestrator/orchestrator.py
+++ b/evaluation/tau2-bench/tau2/orchestrator/orchestrator.py
@@ -5,6 +5,7 @@ from datetime import datetime, timedelta
 from enum import Enum
 from typing import Any, Optional
 import os
+import requests
 from loguru import logger
 
 def _log_with_time(msg: str, level: str = "INFO"):
@@ -80,6 +81,8 @@ class Orchestrator:
         self.cur_transfer_dir = cur_transfer_dir
         self.use_model_tool = use_model_tool
         self.model_config_path = model_config_path
+        self.job_id = f"tau2:{domain}:{task.id}:{uuid.uuid4().hex[:8]}"
+        self.router_released = False
         self.agent_state: Optional[Any] = None
         self.user_state: Optional[UserState] = None
         self.trajectory: list[Message] = []
@@ -93,6 +96,33 @@ class Orchestrator:
         self.to_role: Optional[Role] = None
         self.message: Optional[Message] = None
 
+    @staticmethod
+    def _set_llm_router_args(target: Any, job_id: str, is_last_step: bool) -> None:
+        llm_args = getattr(target, "llm_args", None)
+        if isinstance(llm_args, dict):
+            llm_args["job_id"] = job_id
+            llm_args["is_last_step"] = is_last_step
+
+    def _release_router_job(self) -> None:
+        if self.router_released:
+            return
+        router_url = os.getenv("ROUTER_URL")
+        if not router_url:
+            return
+        base_url = router_url.rstrip("/")
+        try:
+            resp = requests.post(
+                f"{base_url}/programs/release",
+                json={"job_id": self.job_id},
+                timeout=2.0,
+            )
+            resp.raise_for_status()
+            self.router_released = True
+        except Exception as exc:
+            logger.warning(
+                f"[Orchestrator] Failed to release job_id={self.job_id} via router: {exc}"
+            )
+
     def initialize(self):
         """
         Initialize the orchestrator.
@@ -260,6 +290,7 @@ class Orchestrator:
                 self.from_role = Role.AGENT
                 self.to_role = Role.USER
             else:
+                self._set_llm_router_args(self.agent, self.job_id, False)
                 first_message, agent_state = self.agent.generate_next_message(
                     None, self.agent_state
                 )
@@ -270,6 +301,7 @@ class Orchestrator:
                 self.done = self.agent.is_stop(first_message)
                 if self.done:
                     self.termination_reason = TerminationReason.AGENT_STOP
+                    self._release_router_job()
 
         self.environment.sync_tools()
         _log_with_time(f"[Orchestrator] Initialization completed in {time.perf_counter() - init_start:.3f}s, solo_mode={self.solo_mode}")
@@ -284,43 +316,46 @@ class Orchestrator:
         _log_with_time(f"[Orchestrator] ========== Starting simulation run for task: {self.task.id} ==========")
         start_time = get_now()
         start = time.perf_counter()
-        self.initialize()
-        _log_with_time(f"[Orchestrator] Starting main loop, max_steps={self.max_steps}, max_errors={self.max_errors}")
-        while not self.done:
-            # with open(os.path.join(self.cur_transfer_dir,f"tau_loop_{self.step_count}"),'w') as f:
-            #     f.write(f"263, tau self.step_count, {self.step_count}")
-            # print(263,'Step',self.step_count)
-            self.step()
-            if self.step_count >= self.max_steps:
-                self.done = True
-                self.termination_reason = TerminationReason.MAX_STEPS
-            if self.num_errors >= self.max_errors:
-                self.done = True
-                self.termination_reason = TerminationReason.TOO_MANY_ERRORS
-        # print(279,'finish all steps')
-        duration = time.perf_counter() - start
-        _log_with_time(f"[Orchestrator] Simulation loop finished: total_steps={self.step_count}, termination_reason={self.termination_reason}, duration={duration:.3f}s")
-        messages = self.get_trajectory()
-        res = get_cost(messages)
-        if res is None:
-            agent_cost, user_cost = None, None
-        else:
-            agent_cost, user_cost = res
-        simulation_run = SimulationRun(
-            id=str(uuid.uuid4()),
-            task_id=self.task.id,
-            start_time=start_time,
-            end_time=get_now(),
-            duration=duration,
-            termination_reason=self.termination_reason.value,
-            reward_info=None,
-            user_cost=user_cost,
-            agent_cost=agent_cost,
-            messages=messages,
-            seed=self.seed,
-        )
-        _log_with_time(f"[Orchestrator] ========== Simulation completed for task: {self.task.id}, total_duration={duration:.3f}s, total_steps={self.step_count}, messages={len(messages)} ==========")
-        return simulation_run
+        try:
+            self.initialize()
+            _log_with_time(f"[Orchestrator] Starting main loop, max_steps={self.max_steps}, max_errors={self.max_errors}")
+            while not self.done:
+                # with open(os.path.join(self.cur_transfer_dir,f"tau_loop_{self.step_count}"),'w') as f:
+                #     f.write(f"263, tau self.step_count, {self.step_count}")
+                # print(263,'Step',self.step_count)
+                self.step()
+                if self.step_count >= self.max_steps:
+                    self.done = True
+                    self.termination_reason = TerminationReason.MAX_STEPS
+                if self.num_errors >= self.max_errors:
+                    self.done = True
+                    self.termination_reason = TerminationReason.TOO_MANY_ERRORS
+            # print(279,'finish all steps')
+            duration = time.perf_counter() - start
+            _log_with_time(f"[Orchestrator] Simulation loop finished: total_steps={self.step_count}, termination_reason={self.termination_reason}, duration={duration:.3f}s")
+            messages = self.get_trajectory()
+            res = get_cost(messages)
+            if res is None:
+                agent_cost, user_cost = None, None
+            else:
+                agent_cost, user_cost = res
+            simulation_run = SimulationRun(
+                id=str(uuid.uuid4()),
+                task_id=self.task.id,
+                start_time=start_time,
+                end_time=get_now(),
+                duration=duration,
+                termination_reason=self.termination_reason.value,
+                reward_info=None,
+                user_cost=user_cost,
+                agent_cost=agent_cost,
+                messages=messages,
+                seed=self.seed,
+            )
+            _log_with_time(f"[Orchestrator] ========== Simulation completed for task: {self.task.id}, total_duration={duration:.3f}s, total_steps={self.step_count}, messages={len(messages)} ==========")
+            return simulation_run
+        finally:
+            self._release_router_job()
 
     def step(self):
         """
@@ -344,6 +379,7 @@ class Orchestrator:
         if self.from_role in [Role.AGENT, Role.ENV] and self.to_role == Role.USER:
             _log_with_time(f"[Orchestrator] Step {self.step_count}: Generating user response...")
             user_gen_start = time.perf_counter()
+            self._set_llm_router_args(self.user, self.job_id, False)
             user_msg, self.user_state = self.user.generate_next_message(
                 self.message, self.user_state
             )
@@ -353,6 +389,7 @@ class Orchestrator:
                 self.done = True
                 self.termination_reason = TerminationReason.USER_STOP
                 _log_with_time(f"[Orchestrator] Step {self.step_count}: User sent STOP signal")
+                self._release_router_job()
             self.trajectory.append(user_msg)
             self.message = user_msg
             self.from_role = Role.USER
@@ -367,6 +404,7 @@ class Orchestrator:
         ) and self.to_role == Role.AGENT:
             _log_with_time(f"[Orchestrator] Step {self.step_count}: Generating agent response...")
             agent_gen_start = time.perf_counter()
+            self._set_llm_router_args(self.agent, self.job_id, False)
             agent_msg, self.agent_state = self.agent.generate_next_message(
                 self.message, self.agent_state
             )
@@ -376,6 +414,7 @@ class Orchestrator:
                 self.done = True
                 self.termination_reason = TerminationReason.AGENT_STOP
                 _log_with_time(f"[Orchestrator] Step {self.step_count}: Agent sent STOP signal")
+                self._release_router_job()
             self.trajectory.append(agent_msg)
             self.message = agent_msg
             self.from_role = Role.AGENT
````

**evaluation/tau2-bench/tau2/utils/llm_utils.py**
````diff
diff --git a/evaluation/tau2-bench/tau2/utils/llm_utils.py b/evaluation/tau2-bench/tau2/utils/llm_utils.py
index bb27c1e..28ca198 100644
--- a/evaluation/tau2-bench/tau2/utils/llm_utils.py
+++ b/evaluation/tau2-bench/tau2/utils/llm_utils.py
@@ -420,6 +420,9 @@ def generate(
     if kwargs.get("num_retries") is None:
         kwargs["num_retries"] = DEFAULT_MAX_RETRIES
 
+    job_id = kwargs.get("job_id")
+    is_last_step = kwargs.get("is_last_step")
+
     if model.startswith("claude") and not ALLOW_SONNET_THINKING:
         kwargs["thinking"] = {"type": "disabled"}
     if role=='assistant':
@@ -449,7 +452,7 @@ def generate(
         if 'nemotron-ultra' in model.lower() or 'nemotron-super' in model.lower():
             response = get_llm_response(model=model,messages=updated_messages,tools=updated_tools,return_raw_response=True,temperature=1,model_type='nv/dev',max_length=8000)
         else:
-            response = get_llm_response(model=model,messages=updated_messages,tools=updated_tools,return_raw_response=True,temperature=1,model_config=model_config,model_config_path=model_config_path,model_config_idx=config_idx,model_type='vllm',max_length=8000)
+            response = get_llm_response(model=model,messages=updated_messages,tools=updated_tools,return_raw_response=True,temperature=1,model_config=model_config,model_config_path=model_config_path,model_config_idx=config_idx,model_type='vllm',max_length=8000,job_id=job_id,is_last_step=is_last_step)
         mode_to_call = None
         tool_calls = []
         input_tokens = 0
@@ -501,7 +504,7 @@ def generate(
                     model_config = json.load(f)[mode_to_call]
                 tools_length = len(tokenizer(str(original_tools))['input_ids'])
                 cut_messages = cut_middle_turns(tokenizer=tokenizer,messages=litellm_messages,max_length=23000-tools_length)
-                response = get_llm_response(model=mode_to_call,messages=cut_messages,tools=original_tools,return_raw_response=True,model_config=model_config,model_config_path=model_config_path,model_config_idx=config_idx,model_type='vllm',max_length=8000)
+                response = get_llm_response(model=mode_to_call,messages=cut_messages,tools=original_tools,return_raw_response=True,model_config=model_config,model_config_path=model_config_path,model_config_idx=config_idx,model_type='vllm',max_length=8000,job_id=job_id,is_last_step=is_last_step)
             else:
                 raise ValueError(f'Model {mode_to_call} is not supported')
             if isinstance(response,str):
````