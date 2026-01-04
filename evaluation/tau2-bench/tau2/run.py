import json
import multiprocessing
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
import os
from loguru import logger
import time
from tau2.utils.logging_config import (
    configure_logging,
    set_task_context,
    clear_task_context,
    get_tau2_logger,
)
from tau2.agent.llm_agent import LLMAgent, LLMGTAgent, LLMSoloAgent
from tau2.data_model.simulation import (
    AgentInfo,
    Info,
    Results,
    RunConfig,
    SimulationRun,
    UserInfo,
)
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment, EnvironmentInfo
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.metrics.agent_metrics import compute_metrics
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.registry import RegistryInfo, registry
from tau2.user.user_simulator import DummyUser, get_global_user_sim_guidelines
from tau2.utils.display import ConsoleDisplay
from tau2.utils.pydantic_utils import get_pydantic_hash
from tau2.utils.utils import DATA_DIR, get_commit_hash, get_now, show_dict_diff


def get_options() -> RegistryInfo:
    """
    Returns options for the simulator.
    """
    return registry.get_info()


def get_environment_info(
    domain_name: str, include_tool_info: bool = False
) -> EnvironmentInfo:
    """Get information about the environment for a registered Domain"""
    global registry
    env_constructor = registry.get_env_constructor(domain_name)
    # print(44,env_constructor)
    # exit(0)
    return env_constructor().get_info(include_tool_info=include_tool_info)


def load_tasks(task_set_name: str, task_path: str, save_to) -> list[Task]:
    """
    Loads the tasks for the given domain.
    """
    global registry
    task_loader = registry.get_tasks_loader(task_set_name)
    # Domain task loaders are not consistent across domains:
    # - Some accept (task_path, save_to) (e.g., retail/telecom/airline)
    # - Some accept no args and use built-in paths (e.g., mock)
    # For local runs we pass an explicit `task_path`; fall back to generic JSON loading if needed.
    if task_path:
        try:
            return task_loader(task_path=task_path, save_to=save_to)
        except TypeError:
            try:
                return task_loader(task_path, save_to)
            except TypeError:
                with open(task_path, "r") as fp:
                    tasks = json.load(fp)
                return [Task.model_validate(task) for task in tasks]
    return task_loader()


def get_tasks(
    task_set_name: str,
    task_ids: Optional[list[str]] = None,
    num_tasks: Optional[int] = None,
    task_path = '',
    save_to = ''
) -> list[Task]:
    """
    Loads the tasks for the given domain.
    """
    # If no explicit task IDs are provided, load the full task list for the domain
    # and (optionally) restrict to the first `num_tasks` for quick eval/debug.
    if task_ids is None:
        tasks = load_tasks(task_set_name=task_set_name, task_path=task_path, save_to=save_to)
        if num_tasks is not None:
            tasks = tasks[:num_tasks]
        return tasks
    tasks = [
        task for task in load_tasks(task_set_name=task_set_name, task_path=task_path, save_to=save_to) if task.id in task_ids
    ]
    if len(tasks) != len(task_ids):
        missing_tasks = set(task_ids) - set([task.id for task in tasks])
        raise ValueError(
            f"Not all tasks were found for task set {task_set_name}: {missing_tasks}"
        )
    if num_tasks is not None:
        tasks = tasks[:num_tasks]
    return tasks


def make_run_name(config: RunConfig) -> str:
    """
    Make a run name from the run config
    """
    clean_llm_agent_name = config.llm_agent.split("/")[-1]
    agent_name = f"{config.agent}_{clean_llm_agent_name}"

    clean_llm_user_name = config.llm_user.split("/")[-1]
    user_name = f"{config.user}_{clean_llm_user_name}"

    return f"{get_now()}_{config.domain}_{agent_name}_{user_name}"


def run_domain(config: RunConfig) -> Results:
    """
    Run simulations for a domain
    """
    config.validate()
    # ConsoleDisplay.display_run_config(config)
    if config.task_set_name is None:
        task_set_name = config.domain
    else:
        task_set_name = config.task_set_name
    tasks = get_tasks(task_set_name, config.task_ids, config.num_tasks, task_path=config.task_path, save_to=config.output_file)
    # print(104,'tasks',tasks)
    # exit(0)
    # 104 config.agent llm_agent
    if "gt" in config.agent:
        total_num_tasks = len(tasks)
        tasks = [task for task in tasks if LLMGTAgent.check_valid_task(task)]
        num_tasks = len(tasks)
        ConsoleDisplay.console.print(
            f"[bold green]Running {num_tasks} out of {total_num_tasks} tasks for GT agent.[/bold green]"
        )
    if "solo" in config.agent:
        total_num_tasks = len(tasks)
        tasks = [task for task in tasks if LLMSoloAgent.check_valid_task(task)]
        num_tasks = len(tasks)
        ConsoleDisplay.console.print(
            f"[bold green]Running {num_tasks} out of {total_num_tasks} tasks for solo agent.[/bold green]"
        )

    num_trials = config.num_trials
    save_to = config.save_to
    if save_to is None:
        save_to = make_run_name(config)
    save_to = config.output_file
    # print('save path:',save_to)
    simulation_results = run_tasks(
        domain=config.domain,
        tasks=tasks,
        agent=config.agent,
        user=config.user,
        llm_agent=config.llm_agent,
        llm_args_agent=config.llm_args_agent,
        llm_user=config.llm_user,
        llm_args_user=config.llm_args_user,
        num_trials=num_trials,
        max_steps=config.max_steps,
        max_errors=config.max_errors,
        save_to=save_to,
        console_display=True,
        evaluation_type=EvaluationType.ALL,
        max_concurrency=config.max_concurrency,
        seed=config.seed,
        log_level=config.log_level,
        cur_transfer_dir=config.cur_transfer_dir,
        model_config_path=config.model_config_path,
        use_model_tool=config.use_model_tool
    )
    # metrics = compute_metrics(simulation_results)
    # ConsoleDisplay.display_agent_metrics(metrics)

    return simulation_results


def run_tasks(
    domain: str,
    tasks: list[Task],
    agent: str,
    user: str,
    llm_agent: Optional[str] = None,
    llm_args_agent: Optional[dict] = None,
    llm_user: Optional[str] = None,
    llm_args_user: Optional[dict] = None,
    num_trials: int = 1,
    max_steps: int = 100,
    max_errors: int = 10,
    save_to: Optional[str | Path] = None,
    console_display: bool = True,
    evaluation_type: EvaluationType = EvaluationType.ALL,
    max_concurrency: int = 1,
    seed: Optional[int] = 300,
    log_level: Optional[str] = "INFO",
    cur_transfer_dir: str = '',
    model_config_path: str = '',
    use_model_tool: bool = False
) -> Results:
    """
    Runs tasks for a given domain.
    If llm_as_judge is True, the LLM will be used to annotate the simulation run.
    Calculates the reward for the simulation run.
    Args:
        domain (str): The domain to run the simulation on.
        tasks (list[Task]): The tasks to run.
        agent (str): The agent to run the simulation on.
        user (str): The user to run the simulation on.
        llm_agent (str): The model to use for the agent.
        llm_args_agent (dict): The arguments to pass to the LLM for the agent.
        llm_user (str): The model to use for the user.
        llm_args_user (dict): The arguments to pass to the LLM for the user.
        max_steps (int): The maximum number of steps to run the simulation.
        max_errors (int): The maximum number of errors to allow in the simulation.
        save_to (str | Path): The path to json file where to save the simulation results. If the file already exists, it will try to resume the run.
        evaluation_type (EvaluationType): The type of evaluation to use.
        max_concurrency (int): The maximum number of concurrent simulations to run.
        seed (int): The seed to use for the simulation.
        log_level (str): The log level to use.
    Returns:
        The simulation results and the annotations (if llm_review is True).
    """
    if isinstance(save_to, str):
        save_to = Path(save_to)
    # save_to is optional for tests / quick runs. When unset, disable on-disk checkpointing.
    # We also accept a directory path (e.g. ".") for backwards-compatible callers.
    save_dir: Optional[str]
    if save_to is None:
        save_dir = None
    else:
        # Directory path -> no checkpointing; just use it as an output dir if needed.
        if isinstance(save_to, Path) and save_to.suffix != ".json":
            save_dir = str(save_to)
            save_to = None
        else:
            save_dir = str(save_to)
            assert save_dir.endswith(".json")
            save_dir = save_dir[: -len(".json")]
    # updated_tasks = []

    # Configure tau2 logging with the specified level.
    # If TAU2_LOG_FILE is set (e.g., by run_local.py), also append structured logs to that file.
    configure_logging(level=log_level, file_handler=os.getenv("TAU2_LOG_FILE") or None)
    tau2_logger = get_tau2_logger()
    
    # Set log level for loguru logger as well
    logger.remove()
    # Loguru does not know about our custom PROFILE/USER_JUDGE names.
    # Map them to a valid loguru level to avoid crashing when --log-level PROFILE is used.
    loguru_level = log_level
    if isinstance(log_level, str) and log_level.upper() in ("PROFILE", "USER_JUDGE"):
        loguru_level = "INFO"
    logger.add(lambda msg: print(msg), level=loguru_level)
    
    if len(tasks) == 0:
        raise ValueError("No tasks to run")
    if num_trials <= 0:
        raise ValueError("Number of trials must be greater than 0")
    if max_steps <= 0:
        raise ValueError("Max steps must be greater than 0")
    if max_errors <= 0:
        raise ValueError("Max errors must be greater than 0")

    random.seed(seed)

    seeds = [random.randint(0, 1000000) for _ in range(num_trials)]
    if "seed" in llm_args_agent:
        logger.warning("Each trial will modify the seed for the agent")

    if "seed" in llm_args_user:
        logger.warning("Each trial will modify the seed for the user")

    lock = multiprocessing.Lock()

    info = get_info(
        domain=domain,
        agent=agent,
        user=user,
        llm_agent=llm_agent,
        llm_args_agent=llm_args_agent,
        llm_user=llm_user,
        llm_args_user=llm_args_user,
        num_trials=num_trials,
        max_steps=max_steps,
        max_errors=max_errors,
        seed=seed,
    )
    simulation_results = Results(
        info=info,
        tasks=tasks,
        simulations=[],
    )
    done_runs = set()
    if save_dir is not None and not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if save_to is not None:
        # If save_to already exists, check if the user wants to resume the run.
        if save_to.exists():
            os.remove(save_to)
        if not save_to.parent.exists():
            save_to.parent.mkdir(parents=True, exist_ok=True)
        # with open(save_to, "w") as fp:
        #     fp.write(simulation_results.model_dump_json(indent=2))

    def _save(simulation: SimulationRun,latency):
        # print(268,'save')
        if save_to is None or save_dir is None:
            return
        cur_simulation = simulation.model_dump()
        with open(os.path.join(save_dir,cur_simulation['id']+'.json'),'w') as f:
            json.dump({
                'reward': cur_simulation["reward_info"]['reward'],
            },f,indent=2)
            

    def _run(task: Task, trial: int, seed: int, progress_str: str) -> SimulationRun:
        # Set task context for logging in this thread
        set_task_context(task.id, domain)
        
        try:
            start_time = time.time()
            simulation = run_task(
                domain=domain,
                task=task,
                agent=agent,
                user=user,
                llm_agent=llm_agent,
                llm_args_agent=llm_args_agent,
                llm_user=llm_user,
                llm_args_user=llm_args_user,
                max_steps=max_steps,
                max_errors=max_errors,
                evaluation_type=evaluation_type,
                seed=seed,
                cur_transfer_dir=cur_transfer_dir,
                model_config_path=model_config_path,
                use_model_tool=use_model_tool
            )
            latency = time.time()-start_time
            simulation.trial = trial
            _save(simulation,latency=latency)
            return simulation
        finally:
            # Clear task context when done
            clear_task_context()

    def _emit_task_complete_marker(
        *,
        domain: str,
        trial: int,
        task_id: str,
        status: str,
        completed_domain: int,
        total_domain: int,
    ) -> None:
        # Machine-parsable marker for run_local.py to compute overall ETA.
        print(
            "[TAU2_TASK_COMPLETE] "
            f"domain={domain} trial={trial} task_id={task_id} status={status} "
            f"completed_domain={completed_domain} total_domain={total_domain}",
            flush=True,
        )

    args = []
    total_domain = len(tasks) * num_trials
    completed_domain = 0
    for trial in range(num_trials):
        print(f"Starting trial {trial+1}/{num_trials}", flush=True)
        for i, task in enumerate(tasks):
            print(f"Processing task {i+1}/{len(tasks)}: {task.id} (trial {trial+1})", flush=True)
            if (trial, task.id, seeds[trial]) in done_runs:
                ConsoleDisplay.console.print(
                    f"[bold yellow]Skipping task {task.id}, trial {trial} because it has already been run.[/bold yellow]"
                )
                completed_domain += 1
                _emit_task_complete_marker(
                    domain=domain,
                    trial=trial,
                    task_id=task.id,
                    status="skipped",
                    completed_domain=completed_domain,
                    total_domain=total_domain,
                )
                continue
            progress_str = f"{i}/{len(tasks)} (trial {trial + 1}/{num_trials})"
            args.append((task, trial, seeds[trial], progress_str))

    tau2_logger.info(
        f"Starting evaluation: scheduled={len(args)} total_domain={total_domain} "
        f"max_concurrency={max_concurrency}"
    )

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {executor.submit(_run, *arg): arg for arg in args}

        for future in as_completed(futures):
            task, trial, _seed, _progress_str = futures[future]
            status = "ok"
            try:
                result = future.result()
                if result:
                    simulation_results.simulations.append(result)
            except Exception as e:
                status = "error"
                tau2_logger.warning(f"Task failed with error: {e}")
            finally:
                completed_domain += 1
                _emit_task_complete_marker(
                    domain=domain,
                    trial=trial,
                    task_id=task.id,
                    status=status,
                    completed_domain=completed_domain,
                    total_domain=total_domain,
                )
    
    tau2_logger.info(f"Completed {len(simulation_results.simulations)} simulations")
    # ConsoleDisplay.console.print(
    #     "\n✨ [bold green]Successfully completed all simulations![/bold green]\n"
    #     "To review the simulations, run: [bold blue]tau2 view[/bold blue]"
    # )
    return simulation_results


def run_task(
    domain: str,
    task: Task,
    agent: str,
    user: str,
    llm_agent: Optional[str] = None,
    llm_args_agent: Optional[dict] = None,
    llm_user: Optional[str] = None,
    llm_args_user: Optional[dict] = None,
    max_steps: int = 100,
    max_errors: int = 10,
    evaluation_type: EvaluationType = EvaluationType.ALL,
    seed: Optional[int] = None,
    cur_transfer_dir: str = '',
    model_config_path: str = '',
    use_model_tool: bool = False
) -> SimulationRun:
    """
    Runs tasks for a given domain.
     If llm_as_judge is True, the LLM will be used to annotate the simulation run.
     Calculates the reward for the simulation run.
     Args:
         domain (str): The domain to run the simulation on.
         task (Task): The task to run.
         agent (str): The agent to run the simulation on.
         user (str): The user to run the simulation on.
         llm_agent (str): The model to use for the agent.
         llm_args_agent (dict): The arguments to pass to the LLM for the agent.
         llm_user (str): The model to use for the user.
         llm_args_user (dict): The arguments to pass to the LLM for the user.
         max_steps (int): The maximum number of steps to run the simulation.
         max_errors (int): The maximum number of errors to allow in the simulation.
         evaluation_type (EvaluationType): The type of evaluation to use.
         seed (int): The seed to use for the simulation.
     Returns:
         The simulation run.
    """

    if max_steps <= 0:
        raise ValueError("Max steps must be greater than 0")
    if max_errors <= 0:
        raise ValueError("Max errors must be greater than 0")
    # if not os.path.isdir(cur_transfer_dir):
    #     os.makedirs(cur_transfer_dir,exist_ok=True)
    global registry
    logger.info(
        f"STARTING SIMULATION: Domain: {domain}, Task: {task.id}, Agent: {agent}, User: {user}"
    )
    environment_constructor = registry.get_env_constructor(domain)
    environment = environment_constructor()
    AgentConstructor = registry.get_agent_constructor(agent)

    solo_mode = False
    if issubclass(AgentConstructor, LLMAgent):
        # agent class here
        # print(395,environment)
        # exit(0)
        agent = AgentConstructor(
            tools=environment.get_tools(),
            domain_policy=environment.get_policy(),
            llm=llm_agent,
            llm_args=llm_args_agent,
            cur_transfer_dir=cur_transfer_dir,
            use_model_tool=use_model_tool,
            model_config_path=model_config_path,
            domain=domain
        )
    elif issubclass(AgentConstructor, LLMGTAgent):
        agent = AgentConstructor(
            tools=environment.get_tools(),
            domain_policy=environment.get_policy(),
            llm=llm_agent,
            llm_args=llm_args_agent,
            task=task,
        )
    elif issubclass(AgentConstructor, LLMSoloAgent):
        solo_mode = True
        environment: Environment = environment_constructor(solo_mode=True)
        user_tools = environment.get_user_tools() if environment.user_tools else []
        agent = AgentConstructor(
            tools=environment.get_tools() + user_tools,
            domain_policy=environment.get_policy(),
            llm=llm_agent,
            llm_args=llm_args_agent,
            task=task,
        )
    else:
        raise ValueError(
            f"Unknown agent type: {AgentConstructor}. Should be LLMAgent or LLMSoloAgent"
        )
    try:
        user_tools = environment.get_user_tools()
    except Exception:
        user_tools = None

    UserConstructor = registry.get_user_constructor(user)
    if issubclass(UserConstructor, DummyUser):
        assert isinstance(agent, LLMSoloAgent), (
            "Dummy user can only be used with solo agent"
        )

    user = UserConstructor(
        tools=user_tools,
        instructions=str(task.user_scenario),
        llm=llm_user,
        llm_args=llm_args_user,
    )

    orchestrator = Orchestrator(
        domain=domain,
        agent=agent,
        user=user,
        environment=environment,
        task=task,
        max_steps=max_steps,
        max_errors=max_errors,
        seed=seed,
        solo_mode=solo_mode,
        cur_transfer_dir=cur_transfer_dir,
        model_config_path=model_config_path,
        use_model_tool=use_model_tool,
    )
    simulation = orchestrator.run()
    # print(472,'after run')

    reward_info = evaluate_simulation(
        domain=domain,
        task=task,
        simulation=simulation,
        evaluation_type=evaluation_type,
        solo_mode=solo_mode,
    )
    # print(481,'after eval')

    simulation.reward_info = reward_info
    # print(495,reward_info)

    logger.info(
        f"FINISHED SIMULATION: Domain: {domain}, Task: {task.id}, Agent: {agent.__class__.__name__}, User: {user.__class__.__name__}. Reward: {reward_info.reward}"
    )
    return simulation


def get_info(
    domain: str,
    agent: str,
    user: str,
    llm_agent: Optional[str] = None,
    llm_args_agent: Optional[dict] = None,
    llm_user: Optional[str] = None,
    llm_args_user: Optional[dict] = None,
    num_trials: int = 1,
    max_steps: int = 100,
    max_errors: int = 10,
    seed: Optional[int] = None,
) -> Info:
    user_info = UserInfo(
        implementation=user,
        llm=llm_user,
        llm_args=llm_args_user,
        global_simulation_guidelines=get_global_user_sim_guidelines(),
    )
    agent_info = AgentInfo(
        implementation=agent,
        llm=llm_agent,
        llm_args=llm_args_agent,
    )
    environment_info = get_environment_info(
        domain, include_tool_info=False
    )  # NOTE: Not saving tool info to avoid clutter.
    return Info(
        git_commit='none',
        num_trials=num_trials,
        max_steps=max_steps,
        max_errors=max_errors,
        user_info=user_info,
        agent_info=agent_info,
        environment_info=environment_info,
        seed=seed,
    )
