import asyncio
import logging
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

from tasks.ale_bench.prompt import create_prompt
from tasks.ale_bench.task import run_ale_bench_task
from tinker_cookbook.recipes.ttt.state import AleBenchState, State
from tinker_cookbook.recipes.ttt.sampler import StateSampler, get_or_create_sampler_with_default


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

def verify_ale_bench(
    generation: str,
    step: int,
    num_cpus_per_task: int | None = None,
    problem_idx: str | None = None,
    log_dir: str | None = None,
) -> dict:
    logtree.log_text(f"Starting gen")

    out = run_ale_bench_task(generation, problem_id=problem_idx, lite_version=False, log_dir=log_dir, num_cpus_per_task=num_cpus_per_task)

    return {
        "score": out["score"],
        "msg": out["msg"],
        "correctness": out["correctness"],
        "performance": out["performance"],
        "stdout": out.get("stdout", ""),
    }


logger = logging.getLogger(__name__)

T = TypeVar("T")

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


def parse_ale_bench_problem_idx(problem_idx: str) -> tuple[int, str | None]:
    if "_improvement" in problem_idx:
        problem_id = problem_idx.replace("_improvement", "")
        return problem_id, "v1"
    return problem_idx, "v1"


class AleBenchEnv(ProblemEnv):
    def __init__(
        self,
        renderer: renderers.Renderer,
        initial_state: State,
        sampler: StateSampler,
        timeout: float = 1000.0,
        num_cpus_per_task: int = 1,
        problem_idx: str = "v1",
        eval_timeout: int = 300,
        log_path: str = "",
        convo_prefix=None,
    ):
        super().__init__(renderer, convo_prefix=convo_prefix)
        
        if initial_state is None:
            raise ValueError("initial_state is required and cannot be None")
        if sampler is None:
            raise ValueError("sampler is required and cannot be None")
        
        self.timeout = timeout
        self.num_cpus_per_task = num_cpus_per_task
        problem_id, improvement_version = parse_ale_bench_problem_idx(problem_idx)
        self.problem_idx = problem_id
        self.improvement_version = improvement_version
        self.eval_timeout = eval_timeout
        self.log_path = log_path
        self.initial_state = initial_state
        self.sampler = sampler
        self.state = initial_state

    def get_question(self) -> str:
        state = self.initial_state

        assert self.problem_idx in {"ahc039", "ahc058"}
        
        # Use improvement template for iterative optimization
        return self._get_improvement_prompt(state)

    def _get_improvement_prompt(self, state: AleBenchState) -> str:
        """Build contextual improvement prompt with value context."""
        if self.problem_idx == "ahc058":
            target = 6_500_000
        elif self.problem_idx == "ahc039":
            target = 5000
        else:
            raise ValueError(f"Problem ID {self.problem_idx} not supported")

        if state.value > 0:
            current_performance = state.value
            value_ctx = f"\nCurrent performance (higher is better): {current_performance:.4f}"
            value_ctx += f"\nTarget: {target}. Current gap: {target - current_performance:.4f}"
        else:
            value_ctx = f"\nTarget performance: {target}"
        
        prompt = create_prompt(self.problem_idx)
        prompt = prompt.replace("<<<LAST_CODE>>>", state.code if state.code else "# No previous attempt has been made.")
        prompt = prompt.replace("<<<VALUE_CONTEXT>>>", value_ctx)
        return prompt

    def check_format(self, parsed_code: str) -> bool:
        if (parsed_code is None) or (parsed_code.strip() == ''):
            return False
        return True

    async def check_answer(self, parsed_code: str, step: int) -> dict[str, Any]:
        if (parsed_code is None) or (parsed_code.strip() == ''):
            return {
                "score": 0.0,
                "msg": "Invalid code",
                "correctness": 0.0,
                "performance": 0.0,
            }
        # Async grading: this will run verify_ale_bench in a thread, with a timeout,
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
        parsed_code = last_codeblock_postprocess_without_seps(response, ["cpp"])
        correct_format = float(parse_success) and float(self.check_format(parsed_code))

        outs = await self.check_answer(parsed_code, step_idx)
        score = outs["score"]
        correctness = outs["correctness"]
        performance = outs["performance"]
        msg = outs["msg"]

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
                "runtime": performance,
                "score": score,
                "correctness": correctness,
                "correct": correctness,
                "upper_bound": performance,
                "msg": msg,
                "predicted_grid": None,
                "prompt": self.get_question(),
                "response": message['content'],
                "ref": msg,
            },
        )

        # Update sampler with new state if we have valid result (defer save until flush)
        if correctness > 0:
            next_state = AleBenchState(
                timestep=step_idx,
                code=parsed_code,
                value=performance,  # higher = better
            )
            self.sampler.update_states([next_state], [self.initial_state], save=False)

        return step_result

    def get_reference_answer(self) -> str:
        """Return the reference answer for logging purposes."""
        raise NotImplementedError("Reference answer not available for gpu mode.")


SAFE_GRADE_MAX_WORKERS = 4096
SAFE_GRADE_EXECUTOR = ThreadPoolExecutor(max_workers=SAFE_GRADE_MAX_WORKERS)


async def safe_grade(
    given_answer: str,
    step: int,
    timeout: float = 10.0,
    num_cpus_per_task: int = 1,
    problem_idx: str = "ahc039",
    eval_timeout: int = 300,
    log_path: str = "",
    state: AleBenchState = None,
) -> dict[str, Any]:
    """Async grader: runs verify_cp in a background thread with asyncio timeout."""
    loop = asyncio.get_running_loop()
    try:
        out = await asyncio.wait_for(
            loop.run_in_executor(
                SAFE_GRADE_EXECUTOR,
                partial(
                    verify_ale_bench,
                    given_answer,
                    step,
                    num_cpus_per_task=num_cpus_per_task,
                    problem_idx=problem_idx,
                    log_dir=log_path,
                )
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


class AleBenchDataset(RLDataset):
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
    ):
        n, _ = parse_ale_bench_problem_idx(problem_idx)
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
                AleBenchEnv,
                self.renderer,
                initial_state=initial_state,
                sampler=self.sampler,
                num_cpus_per_task=self.num_cpus_per_task,
                problem_idx=self.problem_idx,
                eval_timeout=self.eval_timeout,
                log_path=self.log_path,
                convo_prefix=None,
            ),
            num_envs=group_size,
        )


@chz.chz
class AleBenchDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    seed: int = 0
    problem_idx: str = "26"
    num_cpus_per_task: int = 1
    eval_timeout: int = 300
    dataset_timeout: int = 300
    sampler_type: str = "greedy"
    initial_exp_type: str = "none"  # "best_available", "none", "random"
    log_path: str = ""

    async def __call__(self) -> tuple[AleBenchDataset, AleBenchDataset]:
        if self.problem_idx is None:
            raise ValueError("problem_idx is required")
        if not self.log_path:
            raise ValueError("log_path is required for ale_bench dataset")
        
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        
        n, _ = parse_ale_bench_problem_idx(self.problem_idx)
        sampler = get_or_create_sampler_with_default(
            self.sampler_type, self.log_path, self.problem_idx, initial_exp_type=self.initial_exp_type, n=n, batch_size=self.batch_size,
        )
        
        datasets = [
            AleBenchDataset(
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
            )
            for split in ("train", "test")
        ]
        return (datasets[0], datasets[1])


# Populate the dataset builder map after all classes are defined
DATASET_BUILDER_MAP = {
    "ale_bench": AleBenchDatasetBuilder
}


def get_ale_bench_dataset_builder(
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
) -> RLDatasetBuilder:
    """
    Unified function to get any ale_bench dataset builder.
    Args:
        dataset_name: One of "ale_bench"
        batch_size: Number of groups per batch
        model_name_for_tokenizer: Model name for tokenizer
        renderer_name: Name of the renderer to use
        group_size: Number of environments per group
        problem_idx: "ahc039" or "ahc058"
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
            f"Unknown ale_bench dataset: {dataset_name}. Available: {list(DATASET_BUILDER_MAP.keys())}"
        )
    
    if not log_path:
        raise ValueError("log_path is required for ale_bench dataset")

    # Validate problem_idx format
    try:
        n, _ = parse_ale_bench_problem_idx(problem_idx)
    except:
        raise ValueError(f"Invalid problem_idx: {problem_idx}. Must be v1")

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
    )
    return builder
