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

from tasks.alphaevolve_cp.task import CirclePackingTask
from tasks.alphaevolve_cp.prompt import CP_IMPROVEMENT_TEMPLATE, CP_IMPROVEMENT_TEMPLATE_V2
from tasks.alphaevolve_cp.verifier import validate_packing
from tinker_cookbook.recipes.ttt.state import CirclePackingState, State
from tinker_cookbook.recipes.ttt.sampler import StateSampler, get_or_create_sampler_with_default, SAMPLER_TYPES, INITIAL_EXP_TYPES


def last_codeblock_postprocess(input_text, codeblock_seps=['python', 'cpp', 'java', 'cuda'], last_response_strict=True):
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
        
        return f'```{language}\n{code_content}\n```'
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

def verify_cp(
    generation: str,
    step: int,
    num_cpus_per_task: int = 1,
    problem_idx: int = 26,
    eval_timeout: int = 300,
    log_path: str = "",
    state: CirclePackingState = None,
) -> dict:
    config = default_config.copy()
    config["ttt_rm"] = default_config["ttt_rm"].copy()
    config["ttt_rm"]["num_cpus_per_task"] = num_cpus_per_task
    config["ttt_rm"]["n_item"] = problem_idx
    config["ttt_rm"]["eval_timeout"] = eval_timeout
    config_ns = dict_to_ns(config)

    logtree.log_text(f"Starting gen, {config}")

    task = CirclePackingTask(config_ns, log_path)
    out = task.compute_score(generation, step=step, state=state)

    # Convert [centers, radii, sum] to [[x,y,r], ...] format
    circles = None
    raw_constr = out.get("result_construction")
    if raw_constr and len(raw_constr) >= 2:
        centers, radii = raw_constr[0], raw_constr[1]
        try:
            circles = [[float(centers[i][0]), float(centers[i][1]), float(radii[i])] for i in range(len(radii))]
        except Exception:
            circles = None

    return {
        "sum_radii": out["score"],
        "msg": out["msg"],
        "correctness": out["correctness"],
        "performance": out["performance"],
        "result_construction": circles,
        "stdout": out.get("stdout", ""),
    }


logger = logging.getLogger(__name__)


def parse_cp_problem_idx(problem_idx: str) -> tuple[int, str | None]:
    """Parse problem_idx like '26', '32', '26_improvement', '26_improvement_v2', '26_improvement_state_only' -> (n, improvement_version)"""
    # Strip state_only suffix for version detection, but keep it in the version string
    base_idx = problem_idx
    state_only_suffix = "_state_only" if "_state_only" in problem_idx else ""
    base_idx = base_idx.replace("_state_only", "")
    
    if "_improvement_v2" in base_idx:
        n = int(base_idx.replace("_improvement_v2", ""))
        return n, "v2" + state_only_suffix
    elif "_improvement" in base_idx:
        n = int(base_idx.replace("_improvement", ""))
        return n, "v1" + state_only_suffix
    return int(base_idx), None


def _is_entropic_adv(adv_estimator: str | None) -> bool:
    return adv_estimator in ("entropic", "entropic_adaptive_beta")


class CirclePackingEnv(ProblemEnv):
    def __init__(
        self,
        renderer: renderers.Renderer,
        initial_state: State,
        sampler: StateSampler,
        timeout: float = 1000.0,
        num_cpus_per_task: int = 1,
        problem_idx: str = "26",
        eval_timeout: int = 300,
        log_path: str = "",
        convo_prefix=None,
        adv_estimator: str | None = None,
    ):
        super().__init__(renderer, convo_prefix=convo_prefix)
        
        if initial_state is None:
            raise ValueError("initial_state is required and cannot be None")
        if sampler is None:
            raise ValueError("sampler is required and cannot be None")
        
        self.timeout = timeout
        self.num_cpus_per_task = num_cpus_per_task
        n, improvement_version = parse_cp_problem_idx(problem_idx)
        self.problem_idx = n
        self.improvement_version = improvement_version
        self.eval_timeout = eval_timeout
        self.log_path = log_path
        self.initial_state = initial_state
        self.sampler = sampler
        self.state = initial_state
        self.adv_estimator = adv_estimator

    def get_question(self) -> str:
        return self._get_improvement_prompt(self.initial_state)

    def _get_improvement_prompt(self, state: CirclePackingState) -> str:
        """Build contextual improvement prompt with value context."""
        import inspect
        validator_src = inspect.getsource(validate_packing)
        target = 2.636 if self.problem_idx == 26 else 2.940
        
        # state_only mode: show construction but not code (forces model to write fresh code)
        hide_code = "state_only" in str(self.improvement_version) if self.improvement_version else False
        has_code = state.code and state.code.strip() and not hide_code
        
        # Value context: show before/after if we have parent values
        if state.parent_values and state.value is not None:
            before_sum = state.parent_values[0]
            after_sum = state.value
            value_ctx = f"\nHere are the sum of radii before and after running the code above (higher is better): {before_sum:.6f} -> {after_sum:.6f}"
            value_ctx += f"\nTarget: {target}. Current gap: {target - after_sum:.6f}. Further improvements will also be generously rewarded."
        elif state.value is not None:
            value_ctx = f"\nCurrent sum of radii (higher is better): {state.value:.6f}"
            value_ctx += f"\nTarget: {target}. Current gap: {target - state.value:.6f}. Further improvements will also be generously rewarded."
        else:
            value_ctx = f"\nTarget sum of radii: {target}"
        # Show previous stdout if available
        if state.observation and state.observation.strip():
            stdout = state.observation.strip()
            if len(stdout) > 500:
                stdout = "\n\n\t\t ...(TRUNCATED)...\n" + stdout[-500:]
            value_ctx += f"\n\n--- Previous Program Output ---\n{stdout}\n--- End Output ---"
        
        prompt = CP_IMPROVEMENT_TEMPLATE_V2 if self.improvement_version == "v2" else CP_IMPROVEMENT_TEMPLATE
        prompt = prompt.replace("<<<N>>>", str(self.problem_idx))
        prompt = prompt.replace("<<<VALIDATOR_SRC>>>", validator_src)
        
        if has_code:
            prompt = prompt.replace("<<<LAST_CODE>>>", state.code)
        else:
            prompt = prompt.replace("<<<LAST_CODE>>>", "# No previous code available.")
        
        prompt = prompt.replace("<<<VALUE_CONTEXT>>>", value_ctx)
        return prompt

    def check_format(self, parsed_code: str) -> bool:
        if (parsed_code is None) or (parsed_code.strip() == ''):
            return False
        return True

    async def check_answer(self, parsed_code: str, step: int) -> dict[str, Any]:
        if (parsed_code is None) or (parsed_code.strip() == ''):
            return {
                "sum_radii": 0.0,
                "msg": "Invalid code",
                "correctness": 0.0,
                "performance": 0.0,
                "result_construction": None,
            }
        # Async grading: this will run verify_cp in a thread, with a timeout,
        # without blocking the event loop.

        return await safe_grade(
            parsed_code,
            step,
            self.timeout,
            self.num_cpus_per_task,
            self.problem_idx,
            self.eval_timeout,
            self.log_path,
            state=self.state,
        )

    async def step(self, action: Action, step_idx: int) -> StepResult:
        message, parse_success = self.renderer.parse_response(action)
        response = message["content"]
        parsed_code = last_codeblock_postprocess(response, ["python"])
        correct_format = float(parse_success) and float(self.check_format(parsed_code))

        outs = await self.check_answer(parsed_code, step_idx)
        sum_radii = outs["sum_radii"]
        correctness = outs["correctness"]
        performance = outs["performance"]
        msg = outs["msg"]

        reward = sum_radii if correctness > 0 else 0.0
        
        logtree.log_text(f"Problem: {self.get_question()[:200]}...")
        logtree.log_text(f"Response: {message['content']}")
        logtree.log_text(
            f"Format Valid: {'✓' if correct_format else '✗'}, "
            f"Sum Radii: {sum_radii:.4f}, Performance: {performance:.6f}, Reward: {reward:.4f}, Msg: {msg}"
        )

        step_result = StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "format": correct_format,
                "score": sum_radii,
                "correctness": correctness,
                "correct": correctness,
                "performance": sum_radii if correctness > 0 else None,  # sum of radii, None for incorrect
                "performance/best": sum_radii if correctness > 0 else None,  # higher is better, use max
                "initial_performance": self.initial_state.value,
                "msg": msg,
                "predicted_grid": None,
                "prompt": self.get_question(),
                "response": message['content'],
                "ref": msg,
            },
        )

        # Update sampler with new state if we have valid result (defer save until flush)
        if correctness > 0:
            parent_state = self.initial_state
            parent_values = [parent_state.value] + parent_state.parent_values if parent_state.value is not None else []
            next_state = CirclePackingState(
                timestep=step_idx,
                construction=outs.get("result_construction"),
                code=parsed_code,
                value=sum_radii,  # higher = better
                parent_values=parent_values,
                observation=outs.get("stdout", ""),
            )
            self.sampler.update_states([next_state], [self.initial_state], save=False)
        elif hasattr(self.sampler, 'record_failed_rollout'):
            # Record that we tried this parent but got no valid child (for PUCT visit counts)
            self.sampler.record_failed_rollout(self.initial_state)

        return step_result

    def get_reference_answer(self) -> str:
        """Return the reference answer for logging purposes."""
        raise NotImplementedError("Reference answer not available for circle packing.")


SAFE_GRADE_MAX_WORKERS = 4096
SAFE_GRADE_EXECUTOR = ThreadPoolExecutor(max_workers=SAFE_GRADE_MAX_WORKERS)


async def safe_grade(
    given_answer: str,
    step: int,
    timeout: float = 10.0,
    num_cpus_per_task: int = 1,
    problem_idx: int = 26,
    eval_timeout: int = 300,
    log_path: str = "",
    state: CirclePackingState = None,
) -> dict[str, Any]:
    """Async grader: runs verify_cp in a background thread with asyncio timeout."""
    loop = asyncio.get_running_loop()
    try:
        out = await asyncio.wait_for(
            loop.run_in_executor(
                SAFE_GRADE_EXECUTOR,
                partial(
                    verify_cp,
                    given_answer,
                    step,
                    num_cpus_per_task,
                    problem_idx,
                    eval_timeout,
                    log_path,
                    state,
                )
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(f"Timeout grading {given_answer}")
        return {"sum_radii": 0.0, "msg": "Timeout grading", "correctness": 0.0, "performance": 0.0, "result_construction": None, "stdout": ""}
    except Exception as e:
        logger.warning(f"Exception while grading {given_answer}: {e}")
        return {"sum_radii": 0.0, "msg": f"Error grading: {e}", "correctness": 0.0, "performance": 0.0, "result_construction": None, "stdout": ""}
    return out

class CirclePackingDataset(RLDataset):
    def __init__(
        self,
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
    ):
        n, _ = parse_cp_problem_idx(problem_idx)
        if n not in (26, 32):
            raise ValueError(f"Invalid problem_idx: {problem_idx}. Must be 26, 32, 26_improvement, 32_improvement, 26_improvement_v2, or 32_improvement_v2")
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
                CirclePackingEnv,
                self.renderer,
                initial_state=initial_state,
                sampler=self.sampler,
                num_cpus_per_task=self.num_cpus_per_task,
                problem_idx=self.problem_idx,
                eval_timeout=self.eval_timeout,
                log_path=self.log_path,
                convo_prefix=None,
                adv_estimator=self.adv_estimator,
            ),
            num_envs=group_size,
        )


@chz.chz
class CirclePackingDatasetBuilder(RLDatasetBuilder):
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
    adv_estimator: str | None = None

    async def __call__(self) -> tuple[CirclePackingDataset, CirclePackingDataset]:
        if self.problem_idx is None:
            raise ValueError("problem_idx is required")
        if not self.log_path:
            raise ValueError("log_path is required for CP dataset")
        
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        
        n, _ = parse_cp_problem_idx(self.problem_idx)
        sampler = get_or_create_sampler_with_default(
            self.sampler_type, self.log_path, "cp", initial_exp_type=self.initial_exp_type, n=n, batch_size=self.batch_size, group_size=self.group_size,
        )
        
        datasets = [
            CirclePackingDataset(
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
            )
            for split in ("train", "test")
        ]
        return (datasets[0], datasets[1])


# Populate the dataset builder map after all classes are defined
DATASET_BUILDER_MAP = {
    "cp": CirclePackingDatasetBuilder
}


def get_cp_dataset_builder(
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
) -> RLDatasetBuilder:
    """
    Unified function to get any cp dataset builder.
    Args:
        dataset_name: One of "cp"
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
    if dataset_name not in DATASET_BUILDER_MAP:
        raise ValueError(
            f"Unknown cp dataset: {dataset_name}. Available: {list(DATASET_BUILDER_MAP.keys())}"
        )
    
    if not log_path:
        raise ValueError("log_path is required for CP dataset")

    # Validate problem_idx format
    try:
        n, _ = parse_cp_problem_idx(problem_idx)
        if n not in (26, 32):
            raise ValueError()
    except:
        raise ValueError(f"Invalid problem_idx: {problem_idx}. Must be 26, 32, 26_improvement, 32_improvement, 26_improvement_v2, or 32_improvement_v2")

    builder_class = DATASET_BUILDER_MAP[dataset_name]
    builder = builder_class(
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
    )
    return builder
