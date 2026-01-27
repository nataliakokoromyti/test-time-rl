import asyncio
import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from types import SimpleNamespace
from typing import Any, Literal, Sequence

import chz
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import Action, EnvGroupBuilder, RLDataset, RLDatasetBuilder, StepResult
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logtree

from tasks.erdos_min_overlap.task import ErdosMinOverlapTask
from tinker_cookbook.recipes.ttt.state import ErdosState
from tinker_cookbook.recipes.ttt.sampler import StateSampler
from tinker_cookbook.recipes.ttt.sampler import get_or_create_sampler_with_default, SAMPLER_TYPES, INITIAL_EXP_TYPES


VERIFIER_SRC = '''
import numpy as np


def verify_c5_solution(construction: np.ndarray, c5_achieved: float, n_points: int):
    if not isinstance(construction, np.ndarray):
        try:
            construction = np.array(construction, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert construction to numpy array: {e}")
    
    if len(construction.shape) != 1:
        raise ValueError(f"construction must be 1D array, got shape {construction.shape}")
    
    if construction.shape[0] != n_points:
        raise ValueError(f"Expected construction shape ({n_points},), got {construction.shape}")
    
    if not np.all(np.isfinite(construction)):
        raise ValueError("construction contains NaN or inf values")
    
    if np.any(construction < 0) or np.any(construction > 1):
        raise ValueError(f"h(x) is not in [0, 1]. Range: [{construction.min()}, {construction.max()}]")
    
    n = n_points
    target_sum = n / 2.0
    current_sum = np.sum(construction)
    
    if current_sum != target_sum:
        construction = construction * (target_sum / current_sum)
        if np.any(construction < 0) or np.any(construction > 1):
            raise ValueError(f"After normalization, h(x) is not in [0, 1]. Range: [{construction.min()}, {construction.max()}]")
    
    dx = 2.0 / n_points
    
    j_values = 1.0 - construction
    correlation = np.correlate(construction, j_values, mode="full") * dx
    computed_c5 = np.max(correlation)
    
    if not np.isfinite(computed_c5):
        raise ValueError(f"Computed C5 is not finite: {computed_c5}")
    
    if not np.isclose(computed_c5, c5_achieved, atol=1e-4):
        raise ValueError(f"C5 mismatch: reported {c5_achieved:.6f}, computed {computed_c5:.6f}")
    
    return computed_c5


def evaluate_erdos_solution(construction: np.ndarray, c5_bound: float, n_points: int) -> float:
    verify_c5_solution(construction, c5_bound, n_points)
    return float(c5_bound)
'''


SYSTEM_PROMPT = '''You are an expert in harmonic analysis, numerical optimization, and mathematical discovery.
Your task is to find an improved upper bound for the Erdős minimum overlap problem constant C₅.

## Problem

Find a step function h: [0, 2] → [0, 1] that **minimizes** the overlap integral:

$$C_5 = \\max_k \\int h(x)(1 - h(x+k)) dx$$

**Constraints**:
1. h(x) ∈ [0, 1] for all x
2. ∫₀² h(x) dx = 1

**Discretization**: Represent h as n_points samples over [0, 2].
With dx = 2.0 / n_points:
- 0 ≤ h[i] ≤ 1 for all i
- sum(h) * dx = 1 (equivalently: sum(h) == n_points / 2 exactly)

The evaluation computes: C₅ = max(np.correlate(h, 1-h, mode="full") * dx)

Smaller sequences with less than 1k samples are preferred - they are faster to optimize and evaluate.

**Lower C₅ values are better** - they provide tighter upper bounds on the Erdős constant.

## Budget & Resources
- **Time budget**: <<<BUDGET_S>>>s for your code to run
- **CPUs**: <<<CPUS>>> available

## Rules
- Define `run(seed=42, budget_s=<<<BUDGET_S>>>, **kwargs)` that returns `(h_values, c5_bound, n_points)`
- Use scipy, numpy, cvxpy[CBC,CVXOPT,GLOP,GLPK,GUROBI,MOSEK,PDLP,SCIP,XPRESS,ECOS], math
- Make all helper functions top level, no closures or lambdas
- No filesystem or network IO
- `evaluate_erdos_solution()` and `initial_h_values` (an initial construction, if available) are pre-imported
- Your function must complete within budget_s seconds and return the best solution found

**Lower is better**. Current record: C₅ ≤ 0.38092. Our goal is to find a construction that shows C₅ ≤ 0.38080.'''


def last_codeblock_postprocess(input_text, codeblock_seps=['python', 'cpp', 'java', 'cuda'], last_response_strict=True):
    languages_pattern = '|'.join(map(re.escape, codeblock_seps))
    codeblock_start = f'```({languages_pattern})'
    pattern = re.compile(codeblock_start + r'\n(?!```)(.*?)(?:\n```)?(?=\n```|$)', re.DOTALL)
    matches = list(pattern.finditer(input_text))

    if matches:
        last_match = matches[-1]
        language = last_match.group(1)
        code_content = last_match.group(2).rstrip()
        
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
        "num_cpus_per_task": 2,
        "rew_type": "linear",
        "fail_score": 0.0,
        "eval_timeout": 1100,
        "worst_perf_log": -10000,
        "n_item": 200,
        "sweep_hyperparams": False,
        "max_hyperparam_combos": 1,
    }
}


def verify_erdos(
    generation: str,
    step: int,
    num_cpus_per_task: int = 2,
    eval_timeout: int = 1000,
    log_path: str = "",
    state: ErdosState = None,
) -> dict:
    config = default_config.copy()
    config["ttt_rm"] = default_config["ttt_rm"].copy()
    config["ttt_rm"]["num_cpus_per_task"] = num_cpus_per_task
    config["ttt_rm"]["eval_timeout"] = eval_timeout
    config_ns = dict_to_ns(config)

    logtree.log_text(f"Starting gen, {config}")

    task = ErdosMinOverlapTask(config_ns, log_path)
    out = task.compute_score(generation, step=step, state=state)
    
    construction = None
    actual_c5 = None
    if out["correctness"] > 0 and "result_construction" in out and out["result_construction"] is not None:
        result = out["result_construction"]
        if isinstance(result, (list, tuple)) and len(result) >= 2:
            construction = list(result[0]) if hasattr(result[0], '__iter__') else None
            actual_c5 = float(result[1])  # c5_bound is at index 1
    
    return {
        "score": out["score"],
        "msg": out["msg"],
        "correctness": out["correctness"],
        "performance": -actual_c5 if actual_c5 is not None else None,
        "c5_bound": actual_c5,
        "construction": construction,
        "stdout": out.get("stdout", ""),
    }


logger = logging.getLogger(__name__)
SAFE_GRADE_MAX_WORKERS = 4096
SAFE_GRADE_EXECUTOR = ThreadPoolExecutor(max_workers=SAFE_GRADE_MAX_WORKERS)


def _is_entropic_adv(adv_estimator: str | None) -> bool:
    return adv_estimator in ("entropic", "entropic_adaptive_beta")


class ErdosMinOverlapEnv(ProblemEnv):
    def __init__(
        self,
        renderer: renderers.Renderer,
        initial_state: ErdosState,
        sampler: StateSampler,
        timeout: float = 6000.0,
        num_cpus_per_task: int = 2,
        eval_timeout: int = 1000,
        log_path: str = "",
        convo_prefix=None,
        budget_s: int = 500,
        problem_idx: str = "v1",
        adv_estimator: str | None = None,
    ):
        super().__init__(renderer, convo_prefix=convo_prefix)
        
        if initial_state is None:
            raise ValueError("initial_state is required and cannot be None")
        if sampler is None:
            raise ValueError("sampler is required and cannot be None")
        
        self.timeout = timeout
        self.num_cpus_per_task = num_cpus_per_task
        self.eval_timeout = eval_timeout
        self.log_path = log_path
        self.initial_state = initial_state
        self.sampler = sampler
        self.budget_s = budget_s
        self.state = initial_state
        self.problem_idx = problem_idx
        self.adv_estimator = adv_estimator

    def _truncate_code(self, code: str, max_lines: int = 200) -> str:
        """Keep top and bottom lines, truncate middle."""
        lines = code.split('\n')
        if len(lines) <= max_lines:
            return code
        half = max_lines // 2
        return '\n'.join(lines[:half]) + "\n# ...(middle truncated)...\n" + '\n'.join(lines[-half:])

    def _get_improvement_prompt(self, state: ErdosState) -> str:
        """Build contextual improvement prompt with before/after values and stdout."""
        # state_only mode: show construction but not code (forces model to write fresh code)
        hide_code = "state_only" in self.problem_idx
        has_code = state.code and state.code.strip() and not hide_code
        
        # Value context: show before/after if we have parent values
        # state.value is -c5_bound (higher=better for RL), so negate to get c5_bound (lower=better)
        if state.parent_values and state.value is not None:
            before_bound = -state.parent_values[0]
            after_bound = -state.value
            value_ctx = f"\nHere are the C₅ bounds before and after running the code above (lower is better): {before_bound:.6f} -> {after_bound:.6f}"
            value_ctx += f"\nOur target is to make the C₅ bound tighter, just as a reference, lower it to at least 0.3808."
        elif state.value is not None:
            value_ctx = f"\nCurrent C₅ bound (lower is better): {-state.value:.6f}"
            value_ctx += f"\nOur target is to make the C₅ bound tighter, just as a reference, lower it to at least 0.3808."
        elif state.c5_bound is not None:
            value_ctx = f"\nCurrent C₅ bound (lower is better): {state.c5_bound:.6f}"
            value_ctx += f"\nOur target is to make the C₅ bound tighter, just as a reference, lower it to at least 0.3808."
        else:
            value_ctx = "\nOur target is to make the C₅ bound tighter, just as a reference, lower it to at least 0.3808."
        
        if state.observation and state.observation.strip():
            stdout = state.observation.strip()
            if len(stdout) > 500:
                stdout = "...(truncated)\n" + stdout[-500:]
            value_ctx += f"\n\n--- Previous Program Output ---\n{stdout}\n--- End Output ---"
        
        prompt = SYSTEM_PROMPT
        prompt = prompt.replace("<<<BUDGET_S>>>", str(self.budget_s))
        prompt = prompt.replace("<<<CPUS>>>", str(self.num_cpus_per_task))
        
        h_values_section = ""
        if hasattr(state, 'construction') and state.construction is not None and len(state.construction) > 0:
            h_values_section = f"""
You may want to start your search from the current construction, which you can access through the `initial_h_values` global variable (n={len(state.construction)} samples).
You are encouraged to explore solutions that use other starting points to prevent getting stuck in a local optimum.
"""
        
        # Handle code section
        if has_code:
            clean_code = state.code.strip()
            if clean_code.startswith("```python"):
                clean_code = clean_code[len("```python"):].strip()
            if clean_code.startswith("```"):
                clean_code = clean_code[3:].strip()
            if clean_code.endswith("```"):
                clean_code = clean_code[:-3].strip()
            code_section = f"""
Here is the last code we ran:
```python
{clean_code}
```

You are iteratively optimizing constructions.{value_ctx}

Reason about how you could further improve this construction.
Ideally, try to do something different than the above algorithm. Could be using different algorithmic ideas, adjusting your heuristics, adjusting / sweeping your hyperparemeters, etc. 
Unless you make a meaningful improvement, you will not be rewarded.
"""
        else:
            code_section = f"""
{value_ctx}

Write code to optimize this construction.
"""
        
        return f"""{prompt}
{h_values_section}{code_section}"""

    def get_question(self) -> str:
        """Build prompt from template, injecting previous code from state."""
        return self._get_improvement_prompt(self.initial_state)

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
                "c5_bound": None,
                "construction": None,
            }
        return await safe_grade(
            parsed_code,
            step,
            self.timeout,
            self.num_cpus_per_task,
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
        score = outs["score"]
        correctness = outs["correctness"]
        performance = outs["performance"]
        c5_bound = outs["c5_bound"]
        msg = outs["msg"]

        # Gap-to-goal reward for entropic advantage
        if _is_entropic_adv(self.adv_estimator):
            # Erdos: lower c5_bound is better, target=0.38092
            # performance = -c5_bound (actual bound)
            # reward = goal/current_bound - 1 (inverse because lower is better)
            current_bound = -performance if performance is not None else float('inf')
            reward = 1/current_bound if (correctness > 0 and current_bound > 0) else 0.0
        else:
            reward = score

        # For logging: best_bound is the actual c5 value (only for correct submissions)
        best_bound = c5_bound if correctness > 0 else None

        logtree.log_text(f"Problem: {self.get_question()[:200]}...")
        logtree.log_text(f"Response: {message['content']}")
        logtree.log_text(
            f"Format Valid: {'✓' if correct_format else '✗'}, "
            f"Score: {score:.4f}, BestBound: {best_bound}, Reward: {reward:.4f}, Msg: {msg}"
        )

        step_result = StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "format": correct_format,
                "score": score,
                "correctness": correctness,
                "correct": correctness,
                "c5_bound": c5_bound,
                "performance": best_bound,
                "performance/best": best_bound,
                "initial_performance": -self.initial_state.value if self.initial_state.value is not None else None,
                "msg": msg,
                "predicted_grid": None,
                "prompt": self.get_question(),
                "response": message['content'],
                "ref": msg,
            },
        )

        if correctness > 0 and performance is not None:
            parent_state = self.initial_state
            parent_values = [parent_state.value] + parent_state.parent_values if parent_state.value is not None else []
            construction = outs["construction"]
            next_state = ErdosState(
                timestep=step_idx,
                code=parsed_code,
                value=performance,
                c5_bound=c5_bound,
                construction=construction,
                parent_values=parent_values,
                observation=outs.get("stdout", ""),
            )
            self.sampler.update_states([next_state], [self.initial_state], save=False)
        elif hasattr(self.sampler, 'record_failed_rollout'):
            self.sampler.record_failed_rollout(self.initial_state)

        return step_result

    def get_reference_answer(self) -> str:
        raise NotImplementedError("Reference answer not available for Erdos min overlap.")


async def safe_grade(
    given_answer: str,
    step: int,
    timeout: float = 10.0,
    num_cpus_per_task: int = 2,
    eval_timeout: int = 1000,
    log_path: str = "",
    state: ErdosState = None,
) -> dict[str, Any]:
    assert timeout > 1000, "Outer timeout should be longer than real task timeout."
    logtree.log_text(f"Reached here: {given_answer}")
    loop = asyncio.get_running_loop()
    start_time = time.time()
    try:
        out = await asyncio.wait_for(
            loop.run_in_executor(
                SAFE_GRADE_EXECUTOR,
                partial(
                    verify_erdos,
                    given_answer,
                    step,
                    num_cpus_per_task,
                    eval_timeout,
                    log_path,
                    state,
                )
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        logger.warning(f"Timeout grading: took {elapsed:.1f}s, limit was {timeout:.1f}s")
        return {"score": 0.0, "msg": "Timeout grading", "correctness": 0.0, "performance": 0.0, "c5_bound": None, "construction": None, "stdout": ""}
    except Exception as e:
        logger.warning(f"Exception while grading: {e}")
        return {"score": 0.0, "msg": f"Error grading: {e}", "correctness": 0.0, "performance": 0.0, "c5_bound": None, "construction": None, "stdout": ""}
    return out


class ErdosMinOverlapDataset(RLDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        sampler: StateSampler,
        split: Literal["train", "test"] = "train",
        seed: int = 0,
        num_cpus_per_task: int = 2,
        eval_timeout: int = 1000,
        dataset_timeout: int = 1000,
        log_path: str = "",
        problem_idx: str = "v1",
        adv_estimator: str | None = None,
    ):
        self.split = split
        self.batch_size = batch_size
        self.group_size = group_size if split == "train" else 1
        self.renderer = renderer
        self.num_cpus_per_task = num_cpus_per_task
        self.eval_timeout = eval_timeout
        self.dataset_timeout = dataset_timeout
        self.log_path = log_path
        self.sampler = sampler
        self.problem_idx = problem_idx
        self.adv_estimator = adv_estimator
    
    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        states = self.sampler.sample_states(self.batch_size)
        return [self._make_env_group_builder(state, self.group_size) for state in states]

    def __len__(self) -> int:
        return 1

    def flush(self, step: int | None = None):
        """Flush sampler state to disk. Call after batch completes."""
        self.sampler.flush(step)

    def _make_env_group_builder(
        self, initial_state: ErdosState, group_size: int
    ) -> ProblemGroupBuilder:
        return ProblemGroupBuilder(
            env_thunk=partial(
                ErdosMinOverlapEnv,
                self.renderer,
                initial_state=initial_state,
                sampler=self.sampler,
                num_cpus_per_task=self.num_cpus_per_task,
                eval_timeout=self.eval_timeout,
                log_path=self.log_path,
                convo_prefix=None,
                budget_s=self.dataset_timeout,
                problem_idx=self.problem_idx,
                adv_estimator=self.adv_estimator,
            ),
            num_envs=group_size,
        )


@chz.chz
class ErdosMinOverlapDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    seed: int = 0
    num_cpus_per_task: int = 2
    eval_timeout: int = 1000
    dataset_timeout: int = 1000
    sampler_type: str = "per"
    initial_exp_type: str = "none"
    log_path: str = ""
    problem_idx: str = "v1"
    adv_estimator: str | None = None

    async def __call__(self) -> tuple[ErdosMinOverlapDataset, ErdosMinOverlapDataset]:
        if not self.log_path:
            raise ValueError("log_path is required for Erdos min overlap dataset")
        
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        
        sampler = get_or_create_sampler_with_default(
            self.sampler_type, self.log_path, "erdos", budget_s=self.dataset_timeout, initial_exp_type=self.initial_exp_type, batch_size=self.batch_size, problem_idx=self.problem_idx, group_size=self.group_size,
        )
        
        datasets = [
            ErdosMinOverlapDataset(
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
                problem_idx=self.problem_idx,
                adv_estimator=self.adv_estimator,
            )
            for split in ("train", "test")
        ]
        return (datasets[0], datasets[1])


DATASET_BUILDER_MAP = {
    "erdos": ErdosMinOverlapDatasetBuilder
}


def get_erdos_dataset_builder(
    dataset_path: str,
    dataset_name: str,
    batch_size: int,
    model_name_for_tokenizer: str,
    renderer_name: str,
    group_size: int,
    seed: int = 0,
    num_cpus_per_task: int = 2,
    eval_timeout: int = 1000,
    dataset_timeout: int = 1000,
    sampler_type: str = "per",
    initial_exp_type: str = "none",
    log_path: str = "",
    problem_idx: str = "v1",
    adv_estimator: str | None = None,
) -> RLDatasetBuilder:
    assert dataset_timeout <= eval_timeout, f"Should not set eval timeout less than dataset timeout: dataset_timeout={dataset_timeout}, eval_timeout={eval_timeout}"
    assert dataset_timeout + 200 >= eval_timeout, f"Found eval timeout unexpectedly much higher than dataset timeout: dataset_timeout={dataset_timeout}, eval_timeout={eval_timeout}"
    
    if dataset_name not in DATASET_BUILDER_MAP:
        raise ValueError(f"Unknown erdos dataset: {dataset_name}. Available: {list(DATASET_BUILDER_MAP.keys())}")

    if sampler_type not in SAMPLER_TYPES:
        raise ValueError(f"Unknown sampler_type: {sampler_type}. Supported: {SAMPLER_TYPES}")
    if initial_exp_type not in INITIAL_EXP_TYPES:
        raise ValueError(f"Unknown initial_exp_type: {initial_exp_type}. Supported: {INITIAL_EXP_TYPES}")
    
    if not log_path:
        raise ValueError("log_path is required for Erdos min overlap dataset")

    config_ns = dict_to_ns(default_config)
    ErdosMinOverlapTask(config_ns, log_path)

    builder_class = DATASET_BUILDER_MAP[dataset_name]
    return builder_class(
        batch_size=batch_size,
        model_name_for_tokenizer=model_name_for_tokenizer,
        renderer_name=renderer_name,
        group_size=group_size,
        seed=seed,
        num_cpus_per_task=num_cpus_per_task,
        eval_timeout=eval_timeout,
        dataset_timeout=dataset_timeout,
        sampler_type=sampler_type,
        initial_exp_type=initial_exp_type,
        log_path=log_path,
        problem_idx=problem_idx,
        adv_estimator=adv_estimator,
    )
