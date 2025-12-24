# Tau2-Bench Logging & Diagnostics System

This document details the advanced logging infrastructure and debugging capabilities introduced in the local evaluation runner (`run_local.py`). The system is designed to provide high-resolution performance profiling, cost tracking, and deep introspection into the agentic loop without sacrificing runtime performance.

## 1. Logging Architecture

The logging system is built on top of Python's standard `logging` library but extends it with:
- **Thread-Local Task Context**: Ensures logs from concurrent threads are correctly tagged with their `task_id` and `domain`.
- **Custom Log Levels**: Adds specialized levels for profiling (`PROFILE`) and cost tracking (`USER_JUDGE`).
- **Structured Output**: Uses key-value pairs (e.g., `duration_ms=120.5`) to enable easy parsing by downstream analysis tools.
- **Dual Streaming**: Logs are simultaneously streamed to the console (with `rich` progress bars) and persisted to per-domain log files.

### Log Levels Hierarchy

| Level Name | Value | Description |
|------------|-------|-------------|
| **DEBUG**  | 10    | Deep introspection: raw prompts, tool parameters, state transitions. Verbose. |
| **PROFILE**| 15    | **(New)** Structured timing data for LLM calls, tool execution, and step latency. |
| **USER_JUDGE**| 16 | **(New)** Cost and latency tracking for User Simulator and LLM-as-a-Judge. |
| **INFO**   | 20    | High-level progress, task completion status, and system initialization events. |
| **WARNING**| 30    | Non-critical issues (e.g., retrying a failed API call). |
| **ERROR**  | 40    | Task failures, tool exceptions, or system errors. |

---

## 2. The PROFILE Level (Level 15)

The `PROFILE` level is the core of our performance analysis capability. It emits structured events that can be parsed to reconstruct the timeline of every task.

### Event Schema

All `PROFILE` logs follow this structure:
```text
[PROFILE] YYYY-MM-DD HH:MM:SS.mmm task={task_id} thread={thread_id} type={event_type} {key}={value} ...
```

### Event Types

#### 1. `llm_call`
Recorded when the Agent performs inference (vLLM, OpenAI, etc.).
- **Fields**:
  - `model`: Model name (e.g., `nemotron`, `gpt-4o`).
  - `call_type`: Backend used (`vllm`, `openai`, `claude`).
  - `duration_ms`: Wall-clock time in milliseconds.
  - `has_tool_calls`: Boolean, whether the model invoked tools.
  - `input_tokens` / `output_tokens`: (If available).

#### 2. `tool_call`
Recorded when a local Python tool function is executed.
- **Fields**:
  - `function`: Name of the function called (e.g., `search_products`).
  - `call_type`: Always `local_function`.
  - `duration_ms`: Execution time.
  - `error`: Boolean, true if the tool raised an exception.

#### 3. `expert_call`
Recorded when the Agent routes a query to an Expert Model (Model-as-a-Tool).
- **Fields**:
  - `model`: Expert model name (e.g., `gpt-5`, `Qwen3-32B`).
  - `duration_ms`: Total latency including network overhead.

#### 4. `step_complete`
Recorded at the end of a full agent turn (Thought -> Action -> Observation).
- **Fields**:
  - `step`: The step index (0, 1, 2...).
  - `total_duration_ms`: Total time for the step.
  - `from_role`: Usually `agent`.
  - `to_role`: Usually `env` or `user`.

---

## 3. The USER_JUDGE Level (Level 16)

This level isolates the "meta-costs" of running the benchmark—resources consumed by the evaluation framework itself, rather than the agent.

### Event Types

#### 1. `user_sim`
Calls made by the User Simulator to generate responses.
- **Fields**:
  - `model`: e.g., `gpt-5`.
  - `duration_ms`: Latency.

#### 2. `evaluator`
Calls made by the LLM-as-a-Judge to score the trajectory.
- **Fields**:
  - `model`: e.g., `gpt-4o`.
  - `duration_ms`: Latency.

---

## 4. Debugging Workflows

### Scenario A: High Latency Analysis
**Symptom**: Evaluation takes too long.
**Action**:
1. Run with `--log-level PROFILE`.
2. Extract durations:
   ```bash
   # Average LLM latency
   grep "type=llm_call" eval.log | grep -oP "duration_ms=\d+\.\d+" | cut -d= -f2 | awk '{s+=$1} END {print s/NR}'
   
   # Average Tool latency
   grep "type=tool_call" eval.log | grep -oP "duration_ms=\d+\.\d+" | cut -d= -f2 | awk '{s+=$1} END {print s/NR}'
   ```

### Scenario B: Stuck/Hanging Tasks
**Symptom**: Progress bar stops moving.
**Action**:
1. Check the `logs/tau2_{domain}.log` file (which streams in real-time).
2. Look for the last heartbeat or log entry.
3. Identify the `task_id` and `step` to see if it's stuck in a tool loop or waiting for an API response.

### Scenario C: Tool Execution Errors
**Symptom**: Low success rate.
**Action**:
1. Run a smoke test with `DEBUG` level:
   ```bash
   python run_local.py --domains retail --num-tasks 1 --log-level DEBUG
   ```
2. Search logs for `Traceback` or tool error messages.
3. Verify if the Agent is hallucinating invalid arguments.

---

## 5. Performance Analysis Tools

We provide `analyze_timing_from_tau2_log.py` to automate the parsing of `PROFILE` logs.

### Usage
```bash
python analyze_timing_from_tau2_log.py --log_file logs/eval_retail.log --output_dir analysis_results/
```

### Outputs
- **Latency Distribution Plots**: Histograms of LLM vs. Tool latencies.
- **Cost Analysis**: Token usage breakdown.
- **Trace Visualization**: Gantt-chart style view of selected tasks.

