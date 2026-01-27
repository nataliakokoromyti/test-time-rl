import asyncio
import logging
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from functools import partial
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Literal, Sequence, Tuple, TypeVar

import chz
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import Action, EnvGroupBuilder, RLDataset, RLDatasetBuilder, StepResult
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logtree

from tasks.gpu_mode.task import run_gpu_mode_task
from tasks.gpu_mode.prompt_trimul import TRIMUL_V1_PROMPT, TRIMUL_IMPROVEMENT_TEMPLATE_V0, TRIMUL_IMPROVEMENT_TEMPLATE_V1
from tasks.gpu_mode.prompt_mla_decode import MLA_DECODE_PROMPT_V1, MLA_DECODE_IMPROVEMENT_TEMPLATE_V1

from tinker_cookbook.recipes.ttt.state import GpuModeState, State
from tinker_cookbook.recipes.ttt.sampler import StateSampler, get_or_create_sampler_with_default, SAMPLER_TYPES, INITIAL_EXP_TYPES


# Returns code without ```language``` wrapper
def last_codeblock_postprocess_without_seps(input_text, codeblock_seps=['python', 'cpp', 'java', 'cuda'], last_response_strict=True):
    languages_pattern = '|'.join(map(re.escape, codeblock_seps))
    codeblock_start = f'```({languages_pattern})'
    pattern = re.compile(codeblock_start + r'\n(?!```)(.*?)(?:\n```)?(?=\n```|$)', re.DOTALL)
    matches = list(pattern.finditer(input_text))

    if matches:
        last_match = matches[-1]
        language = last_match.group(1)
        code_content = last_match.group(2).rstrip()
        
        # Check if content is empty
        if not code_content or code_content.strip() == '':
            if last_response_strict:
                return ''
            else:
                return input_text
        
        return code_content
    else:
        if last_response_strict:
            return ''
        else:
            return input_text


def dict_to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_ns(v) for k, v in d.items()})
    return d
    
default_config = {
    "ttt_rm": {
        "num_cpus_per_task": 1,
        "rew_type": "linear",
        "fail_score": 0.0,
        "eval_timeout": 305,
        "n_item": 26,
        "worst_perf_log": 0.0,
        "sweep_hyperparams": False,
        "max_hyperparam_combos": 10,
        "pin_cores": False
    }
}

async def verify_gpu_mode(
    generation: str,
    gpu_type: str,
    task_name: str,
    score_scale: float,
    app_name: str,
) -> dict:
    logtree.log_text(f"Starting gen")

    return await run_gpu_mode_task(generation, gpu_type=gpu_type, task_name=task_name, score_scale=score_scale, app_name=app_name)


logger = logging.getLogger(__name__)

T = TypeVar("T")


def _is_entropic_adv(adv_estimator: str | None) -> bool:
    return adv_estimator in ("entropic", "entropic_adaptive_beta")

def run_with_timeout_signal(
    func: Callable[..., T],
    args: Tuple[Any, ...] = (),
    kwargs: Dict[str, Any] | None = None,
    timeout_seconds: int = 5,
) -> T | None:
    """
    Runs a function with a timeout using ThreadPoolExecutor (cross-platform).

    Args:
        func: The function to execute.
        args: Positional arguments for the function.
        kwargs: Keyword arguments for the function.
        timeout_seconds: Maximum time allowed in seconds.

    Returns:
        The result of the function call, or None if it times out.
    """
    if kwargs is None:
        kwargs = {}

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout_seconds)
        except FuturesTimeoutError:
            logger.warning(f"Function timed out after {timeout_seconds} seconds.")
            result = None
        except Exception as e:
            # Handle other exceptions from the function if needed
            logger.warning(f"Function raised an exception: {e}")
            result = None  # Or re-raise

    return result


def parse_gpu_mode_problem_idx(problem_idx: str) -> tuple[int, str | None]:
    if "_improvement" in problem_idx:
        n = problem_idx.replace("_improvement", "")
        return n, "v1"  # TODO: TriMul uses v0
    return problem_idx, None


class GpuModeEnv(ProblemEnv):
    def __init__(
        self,
        dataset_name: str,
        renderer: renderers.Renderer,
        initial_state: State,
        sampler: StateSampler,
        timeout: float = 2000.0,
        num_cpus_per_task: int = 1,
        problem_idx: str = "v1",
        eval_timeout: int = 300,
        log_path: str = "",
        convo_prefix=None,
        adv_estimator: str | None = None,
        score_scale: float = 3000.0,
    ):
        super().__init__(renderer, convo_prefix=convo_prefix)
        
        if initial_state is None:
            raise ValueError("initial_state is required and cannot be None")
        if sampler is None:
            raise ValueError("sampler is required and cannot be None")
        
        self.dataset_name = dataset_name
        match self.dataset_name:
            case "trimul":
                self.base_prompt = TRIMUL_V1_PROMPT
                if problem_idx == "v1_improvement":
                    self.improvement_prompt = TRIMUL_IMPROVEMENT_TEMPLATE_V1
                elif problem_idx == "v0_improvement":
                    self.improvement_prompt = TRIMUL_IMPROVEMENT_TEMPLATE_V0
                else:
                    raise ValueError(f"Unknown problem index: {problem_idx}")
                self.target = 1000  # Human best is 1371.057μs
                self.gpu_type = "H100"
                self.task_name = "trimul"
                self.score_scale = score_scale
                self.app_name = "discord-bot-runner"
            case "mla_decode_nvidia":
                self.base_prompt = MLA_DECODE_PROMPT_V1
                self.improvement_prompt = MLA_DECODE_IMPROVEMENT_TEMPLATE_V1
                self.target = 1700  # Human best is 1787.457μs (on MI300X)
                self.gpu_type = "H200"
                self.task_name = "mla_decode_nvidia"
                self.score_scale = score_scale
                self.app_name = "discord-bot-runner-mla-decode-nvidia"
            case _:
                raise ValueError(f"Unknown dataset name: {self.dataset_name}")
        
        self.timeout = timeout
        self.num_cpus_per_task = num_cpus_per_task
        n, improvement_version = parse_gpu_mode_problem_idx(problem_idx)
        self.problem_idx = n
        self.improvement_version = improvement_version
        self.eval_timeout = eval_timeout
        self.log_path = log_path
        self.initial_state = initial_state
        self.sampler = sampler
        self.state = initial_state
        self.adv_estimator = adv_estimator

    def get_question(self) -> str:
        state = self.initial_state
        
        # Use improvement template for iterative optimization
        if self.improvement_version:
            return self._get_improvement_prompt(state)
        
        # Default: base prompt with optional previous code
        return self.base_prompt   

    def _get_improvement_prompt(self, state: GpuModeState) -> str:
        """Build contextual improvement prompt with value context."""

        target = self.target

        if state.value < 1_000_000:
            current_runtime = -state.value
            value_ctx = f"\nCurrent runtime (lower is better): {current_runtime:.4f} microseconds"
            value_ctx += f"\nTarget: {target} microseconds. Current gap: {current_runtime - target:.4f} microseconds."
            if self.problem_idx == "V1":
                value_ctx += f"\nIdeally, try to do something different than the above kernel. Could be fusing different operations, could be fusing differently, could be adjusting your heuristics. Unless you make a meaningful improvement, you will not be rewarded."
        else:
            value_ctx = f"\nTarget runtime: {target} microseconds"
        
        prompt = self.improvement_prompt
        prompt = prompt.replace("<<<LAST_CODE>>>", state.code if state.code else "# No previous attempt has been made.")
        prompt = prompt.replace("<<<VALUE_CONTEXT>>>", value_ctx)
        
        return prompt

    def check_format(self, parsed_code: str) -> bool:
        if self.dataset_name == "trimul":
            if (parsed_code is None) or (parsed_code.strip() == '') or ("@triton.jit" not in parsed_code) or ("identity" in parsed_code): # triton check, must write a kernel
                return False
            return True
        
        elif self.dataset_name == "mla_decode_nvidia":
            if (parsed_code is None) or (parsed_code.strip() == '') or ("@triton.jit" not in parsed_code):  # triton check, must write a kernel
                return False
            
            return True
        
        else:
            raise ValueError(f"Unknown dataset name: {self.dataset_name}")

    async def check_answer(self, parsed_code: str, step: int) -> dict[str, Any]:
        if not self.check_format(parsed_code):
            return {
                "score": 0.0,
                "msg": "Invalid code",
                "correctness": 0.0,
                "performance": 0.0,
            }
        # Async grading: this will run verify_gpu_mode in a thread, with a timeout,
        # without blocking the event loop.
        return await safe_grade(
            parsed_code,
            self.gpu_type,
            self.task_name,
            self.score_scale,
            self.app_name,
            self.timeout,
        )

    async def step(self, action: Action, step_idx: int) -> StepResult:
        message, parse_success = self.renderer.parse_response(action)
        response = message["content"]
        parsed_code = last_codeblock_postprocess_without_seps(response, ["python"])
        correct_format = float(parse_success) and float(self.check_format(parsed_code))

        outs = await self.check_answer(parsed_code, step_idx)
        score = outs["score"]
        correctness = outs["correctness"]
        performance = outs["performance"]
        msg = outs["msg"]

        # # For entropic advantage: normalize by target
        # # GPU mode: performance = -runtime, so higher = faster
        # if _is_entropic_adv(self.adv_estimator):
        #     # GPU mode: lower runtime is better, performance = -runtime
        #     # reward = goal/current_runtime - 1 (inverse because lower is better)
        #     # When current_runtime = goal: reward = 0
        #     # When current_runtime < goal: reward > 0 (better/faster)
        #     # When current_runtime > goal: reward < 0 (worse/slower)
        #     goal = 500.0  # target runtime in microseconds
        #     current_runtime = -performance  # actual runtime
        #     reward = 1000/current_runtime if (correctness > 0) else 0.0
        # else:
        #     reward = score

        logtree.log_text(f"Problem: {self.get_question()[:200]}...")
        logtree.log_text(f"Response: {message['content']}")
        logtree.log_text(
            f"Format Valid: {'✓' if correct_format else '✗'}, "
            f"Score: {score:.4f}, Performance: {performance:.6f}, Msg: {msg}"
        )

        step_result = StepResult(
            reward=score,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "format": correct_format,
                "score": score,
                "correctness": correctness,
                "correct": correctness,
                "performance": -performance if correctness > 0 else None,  # runtime in μs, None for incorrect
                "performance/best": -performance if correctness > 0 else None,  # runtime, lower is better, use min
                "initial_performance": -self.initial_state.value if self.initial_state.value is not None else None,
                "msg": msg,
                "predicted_grid": None,
                "prompt": self.get_question(),
                "response": message['content'],
                "ref": msg,
            },
        )

        # Update sampler with new state if we have valid result (defer save until flush)
        if correctness > 0:
            # Combine msg and benchmark_details for observation
            obs_parts = [outs.get("msg", "")]
            if outs.get("benchmark_details"):
                obs_parts.append(f"\nPer-benchmark timing:\n{outs['benchmark_details']}")
            observation = "\n".join(obs_parts)
            next_state = GpuModeState(
                timestep=step_idx,
                code=parsed_code,
                value=performance,  # higher = better, performance is -runtime
                observation=observation,
            )
            self.sampler.update_states([next_state], [self.initial_state], save=False)
        elif hasattr(self.sampler, 'record_failed_rollout'):
            # Record that we tried this parent but got no valid child (for PUCT visit counts)
            self.sampler.record_failed_rollout(self.initial_state)

        return step_result

    def get_reference_answer(self) -> str:
        """Return the reference answer for logging purposes."""
        raise NotImplementedError("Reference answer not available for gpu mode.")


async def safe_grade(
    given_answer: str,
    gpu_type: str,
    task_name: str,
    score_scale: float,
    app_name: str,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Async grader: runs verify_gpu_mode with an asyncio timeout."""
    try:
        out = await asyncio.wait_for(
            verify_gpu_mode(
                given_answer,
                gpu_type,
                task_name,
                score_scale,
                app_name,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(f"Timeout grading {given_answer}")
        return {"score": 0.0, "msg": "Timeout grading", "correctness": 0.0, "performance": 0.0}
    except Exception as e:
        logger.warning(f"Exception while grading {given_answer}: {e}")
        return {"score": 0.0, "msg": f"Error grading: {e}", "correctness": 0.0, "performance": 0.0}
    return out


class GpuModeDataset(RLDataset):
    def __init__(
        self,
        dataset_name: str,
        problem_idx: str,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        sampler: StateSampler,
        split: Literal["train", "test"] = "train",
        seed: int = 0,
        num_cpus_per_task: int = 1,
        eval_timeout: int = 300,
        dataset_timeout: int = 300,
        log_path: str = "",
        adv_estimator: str | None = None,
        score_scale: float = 3000.0,
    ):
        n, _ = parse_gpu_mode_problem_idx(problem_idx)
        self.dataset_name = dataset_name
        self.split = split
        self.batch_size = batch_size
        self.group_size = group_size if split == "train" else 1
        self.renderer = renderer
        self.problem_idx = problem_idx
        self.sampler = sampler
        self.num_cpus_per_task = num_cpus_per_task
        self.eval_timeout = eval_timeout
        self.dataset_timeout = dataset_timeout
        self.log_path = log_path
        self.adv_estimator = adv_estimator
        self.score_scale = score_scale
    
    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        states = self.sampler.sample_states(self.batch_size)
        return [self._make_env_group_builder(state, self.group_size) for state in states]

    def flush(self, step: int | None = None):
        """Flush sampler state to disk. Call after batch completes."""
        self.sampler.flush(step)

    def __len__(self) -> int:
        return 1

    def _make_env_group_builder(
        self, initial_state: State, group_size: int
    ) -> ProblemGroupBuilder:
        return ProblemGroupBuilder(
            env_thunk=partial(
                GpuModeEnv,
                self.dataset_name,
                self.renderer,
                initial_state=initial_state,
                sampler=self.sampler,
                num_cpus_per_task=self.num_cpus_per_task,
                problem_idx=self.problem_idx,
                eval_timeout=self.eval_timeout,
                log_path=self.log_path,
                convo_prefix=None,
                adv_estimator=self.adv_estimator,
                score_scale=self.score_scale,
            ),
            num_envs=group_size,
        )


@chz.chz
class GpuModeDatasetBuilder(RLDatasetBuilder):
    dataset_name: str
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    seed: int = 0
    problem_idx: str = "26"  # "26", "32", "26_improvement", "32_improvement", "26_improvement_v2", "32_improvement_v2"
    num_cpus_per_task: int = 1
    eval_timeout: int = 300
    dataset_timeout: int = 300
    sampler_type: str = "greedy"
    initial_exp_type: str = "none"  # "best_available", "none", "random"
    log_path: str = ""
    score_scale: float = 3000.0,
    adv_estimator: str | None = None

    async def __call__(self) -> tuple[GpuModeDataset, GpuModeDataset]:
        if self.problem_idx is None:
            raise ValueError("problem_idx is required")
        if not self.log_path:
            raise ValueError("log_path is required for gpu_mode dataset")
        
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        
        n, _ = parse_gpu_mode_problem_idx(self.problem_idx)
        sampler = get_or_create_sampler_with_default(
            self.sampler_type, 
            self.log_path, 
            "gpu_mode", 
            initial_exp_type=self.initial_exp_type, 
            n=n, 
            batch_size=self.batch_size, 
            group_size=self.group_size,
        )
        
        datasets = [
            GpuModeDataset(
                dataset_name=self.dataset_name,
                problem_idx=self.problem_idx,
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                sampler=sampler,
                split=split,
                seed=self.seed,
                num_cpus_per_task=self.num_cpus_per_task,
                eval_timeout=self.eval_timeout,
                dataset_timeout=self.dataset_timeout,
                log_path=self.log_path,
                adv_estimator=self.adv_estimator,
                score_scale=self.score_scale,
            )
            for split in ("train", "test")
        ]
        return (datasets[0], datasets[1])


def get_gpu_mode_dataset_builder(
    dataset_path: str,
    dataset_name: str,
    batch_size: int,
    model_name_for_tokenizer: str,
    renderer_name: str,
    group_size: int,
    problem_idx: str,
    seed: int = 0,
    num_cpus_per_task: int = 1,
    eval_timeout: int = 300,
    dataset_timeout: int = 300,
    sampler_type: str = "greedy",
    initial_exp_type: str = "none",
    log_path: str = "",
    adv_estimator: str | None = None,
    gpu_mode_score_scale: float = 3000.0,
) -> RLDatasetBuilder:
    """
    Unified function to get any gpu_mode dataset builder.
    Args:
        dataset_name: One of "gpu_mode"
        batch_size: Number of groups per batch
        model_name_for_tokenizer: Model name for tokenizer
        renderer_name: Name of the renderer to use
        group_size: Number of environments per group
        problem_idx: "26", "32", "26_improvement", "32_improvement", "26_improvement_v2", or "32_improvement_v2"
        seed: Random seed for data shuffling (default: 0)
        num_cpus_per_task: Number of CPUs per task
        eval_timeout: Timeout for evaluation
        dataset_timeout: Timeout for dataset
        sampler_type: Type of experience sampler
        initial_exp_type: Type of initial experience: "best_available", "none", "random"
        log_path: Path for logging and sampler persistence
    Returns:
        The appropriate dataset builder instance
    """
    del dataset_path
    if not log_path:
        raise ValueError("log_path is required for gpu_mode dataset")

    # Validate problem_idx format
    try:
        n, _ = parse_gpu_mode_problem_idx(problem_idx)
    except:
        raise ValueError(f"Invalid problem_idx: {problem_idx}. Must be v1")

    builder_class = GpuModeDatasetBuilder
    builder = builder_class(
        dataset_name=dataset_name,
        batch_size=batch_size,
        model_name_for_tokenizer=model_name_for_tokenizer,
        renderer_name=renderer_name,
        group_size=group_size,
        seed=seed,
        problem_idx=problem_idx,
        num_cpus_per_task=num_cpus_per_task,
        eval_timeout=eval_timeout,
        dataset_timeout=dataset_timeout,
        sampler_type=sampler_type,
        initial_exp_type=initial_exp_type,
        log_path=log_path,
        adv_estimator=adv_estimator,
        score_scale=gpu_mode_score_scale,
    )
    return builder
