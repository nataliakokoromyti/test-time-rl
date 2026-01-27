import asyncio
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from types import SimpleNamespace
from typing import Any, Literal, Sequence

import chz
import numpy as np
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import Action, EnvGroupBuilder, RLDataset, RLDatasetBuilder, StepResult
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logtree

from tasks.denoising.task import DenoisingTask, MAGIC_FUNC, EVALUATE_MSE_FUNC, EVALUATE_POISSON_FUNC
from tasks.denoising.verifier import BASELINES
from tinker_cookbook.recipes.ttt.state import DenoisingState
from tinker_cookbook.recipes.ttt.sampler import StateSampler
from tinker_cookbook.recipes.ttt.sampler import get_or_create_sampler_with_default, SAMPLER_TYPES, INITIAL_EXP_TYPES


SYSTEM_PROMPT = '''You are an expert in computational biology and single-cell RNA-seq analysis.
Your task is to develop a denoising algorithm for scRNA-seq count data. You are experienced in
compuational biology libraries and tools and are familiar with problems in denoising in the single-cell field.

## Problem

Single-cell RNA-seq data is noisy due to technical dropout and low capture efficiency.
Given noisy count data, predict the true expression levels.

Your prediction is evaluated against held-out molecules using two metrics:
1. **MSE** - Mean Squared Error in log-normalized space
2. **Poisson Loss** - Poisson negative log-likelihood

You need to implement a novel denoising algorithm that outperforms the current state-of-the-art without overfitting.

## Data Format

- Input `X`: numpy array of shape (n_cells, n_genes) - **raw count data**
- Output: numpy array of same shape - your denoised counts

## Evaluation

Your output is evaluated using these exact functions:

```python
<<<EVALUATE_MSE_FUNC>>>
```

```python
<<<EVALUATE_POISSON_FUNC>>>
```

## Scoring

**Poisson is a HARD CONSTRAINT.** Your solution is REJECTED if `poisson_norm < 0.97`.
- `poisson_norm = (0.257575 - poisson) / (0.257575 - 0.031739)`
- MAGIC baseline achieves ≈0.97

**Reward = MSE score only** (after passing Poisson constraint).

## Budget & Resources

- **Time budget**: <<<BUDGET_S>>>s for your code to run. You should time your code and make sure it runs within the time budget.
- **CPUs**: <<<CPUS>>> available

## Function Signature to return

```python
def magic_denoise(X, **kwargs):
    # kwargs may include: budget_s, random_state, knn, t, n_pca, solver, decay, knn_max, n_jobs
    # You can add your own parameters too
    # Your implementation
    return denoised_X  # same shape as X
```

## Rules

- Implement `magic_denoise(X, ...)` that returns denoised data
- Use numpy, scipy, sklearn, graphtools, scprep, scanpy
- Make all helper functions top level, no closures or lambdas
- No filesystem or network IO

## Key Insights from Benchmarks

- NORMALIZATION ORDER MATTERS: Denoise raw/log counts first, then normalize. "Reversed normalization order" achieves Poisson ~0.98 vs ~0.55 for standard order.
- Square root transform is variance-stabilizing for Poisson distributions
- Poisson loss is highly affected by low non-zero values - push values < 1 toward zero
- The original MAGIC with reversed normalization achieves best results
'''


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
        "num_cpus_per_task": 4,
        "rew_type": "linear",
        "fail_score": 0.0,
        "eval_timeout": 600,
        "worst_perf_log": -10000,
        "n_item": 200,
        "sweep_hyperparams": False,
        "max_hyperparam_combos": 1,
    }
}


def verify_denoising(
    generation: str,
    step: int,
    num_cpus_per_task: int = 4,
    eval_timeout: int = 500,
    log_path: str = "",
    state: DenoisingState = None,
) -> dict:
    config = default_config.copy()
    config["ttt_rm"] = default_config["ttt_rm"].copy()
    config["ttt_rm"]["num_cpus_per_task"] = num_cpus_per_task
    config["ttt_rm"]["eval_timeout"] = eval_timeout
    config_ns = dict_to_ns(config)

    logtree.log_text(f"Starting denoising gen, {config}")

    task = DenoisingTask(config_ns, log_path)
    out = task.compute_score(generation, step=step, state=state)
    
    # Extract metrics from raw_output
    mse = None
    poisson = None
    mse_normalized = None
    poisson_normalized = None
    
    if out["correctness"] > 0 and out.get("result_construction") is not None:
        raw = out["result_construction"]
        if isinstance(raw, dict):
            mse = raw.get("mse")
            poisson = raw.get("poisson")
        elif isinstance(raw, list) and len(raw) >= 2:
            mse = raw[0] if isinstance(raw[0], (int, float)) else None
            poisson = raw[1] if isinstance(raw[1], (int, float)) else None
        
        baseline = BASELINES["pancreas"]
        mse_range = baseline["baseline_mse"] - baseline["perfect_mse"]
        poisson_range = baseline["baseline_poisson"] - baseline["perfect_poisson"]
        
        if mse is not None and mse_range > 0:
            mse_normalized = (baseline["baseline_mse"] - mse) / mse_range
            mse_normalized = max(0.0, min(1.0, mse_normalized))
        if poisson is not None and poisson_range > 0:
            poisson_normalized = (baseline["baseline_poisson"] - poisson) / poisson_range
            poisson_normalized = max(0.0, min(1.0, poisson_normalized))
    
    return {
        "score": out["score"],
        "msg": out["msg"],
        "correctness": out["correctness"],
        "performance": out.get("performance"),
        "mse": mse,
        "poisson": poisson,
        "mse_normalized": mse_normalized,
        "poisson_normalized": poisson_normalized,
        "stdout": out.get("stdout", ""),
    }


logger = logging.getLogger(__name__)
SAFE_GRADE_MAX_WORKERS = 4096
SAFE_GRADE_EXECUTOR = ThreadPoolExecutor(max_workers=SAFE_GRADE_MAX_WORKERS)


def _is_entropic_adv(adv_estimator: str | None) -> bool:
    return adv_estimator in ("entropic", "entropic_adaptive_beta")


class DenoisingEnv(ProblemEnv):
    def __init__(
        self,
        renderer: renderers.Renderer,
        initial_state: DenoisingState,
        sampler: StateSampler,
        timeout: float = 6000.0,
        num_cpus_per_task: int = 4,
        eval_timeout: int = 500,
        log_path: str = "",
        convo_prefix=None,
        budget_s: int = 300,
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

    def _get_improvement_prompt(self, state: DenoisingState) -> str:
        has_code = state.code and state.code.strip()
        
        value_ctx = ""
        if state.mse is not None or state.poisson is not None:
            metrics = []
            if state.mse is not None:
                metrics.append(f"MSE: {state.mse:.6f}")
            if state.poisson is not None:
                metrics.append(f"Poisson: {state.poisson:.6f}")
            value_ctx = f"\nCurrent metrics (lower is better): {', '.join(metrics)}"
        
        prompt = SYSTEM_PROMPT
        prompt = prompt.replace("<<<BUDGET_S>>>", str(self.budget_s))
        prompt = prompt.replace("<<<CPUS>>>", str(self.num_cpus_per_task))
        prompt = prompt.replace("<<<EVALUATE_MSE_FUNC>>>", EVALUATE_MSE_FUNC)
        prompt = prompt.replace("<<<EVALUATE_POISSON_FUNC>>>", EVALUATE_POISSON_FUNC)
        
        if has_code:
            clean_code = state.code.strip()
            if clean_code.startswith("```python"):
                clean_code = clean_code[len("```python"):].strip()
            if clean_code.startswith("```"):
                clean_code = clean_code[3:].strip()
            if clean_code.endswith("```"):
                clean_code = clean_code[:-3].strip()
            code_section = f"""
Here is the current implementation:
```python
{clean_code}
```

You are iteratively improving the denoising algorithm.{value_ctx}

Reason about how you could improve this approach.
"""
        else:
            code_section = f"""
{value_ctx}

Write code to implement a denoising algorithm.
"""
        
        return f"""{prompt}
{code_section}
Write your improved `magic_denoise` function."""

    def get_question(self) -> str:
        state = self.initial_state
        
        if self.problem_idx == "improvement":
            return self._get_improvement_prompt(state)
        
        prompt = SYSTEM_PROMPT
        prompt = prompt.replace("<<<BUDGET_S>>>", str(self.budget_s))
        prompt = prompt.replace("<<<CPUS>>>", str(self.num_cpus_per_task))
        prompt = prompt.replace("<<<EVALUATE_MSE_FUNC>>>", EVALUATE_MSE_FUNC)
        prompt = prompt.replace("<<<EVALUATE_POISSON_FUNC>>>", EVALUATE_POISSON_FUNC)
        
        value_ctx = ""
        if state.mse is not None or getattr(state, 'poisson', None) is not None:
            metrics = []
            if state.mse is not None:
                metrics.append(f"MSE: {state.mse:.6f}")
            if getattr(state, 'poisson', None) is not None:
                metrics.append(f"Poisson: {state.poisson:.6f}")
            value_ctx = f"Current metrics (lower is better): {', '.join(metrics)}"
        
        previous_attempt = ""
        if state and state.code and state.code.strip() != "":
            clean_code = state.code.strip()
            if clean_code.startswith("```python"):
                clean_code = clean_code[len("```python"):].strip()
            if clean_code.startswith("```"):
                clean_code = clean_code[3:].strip()
            if clean_code.endswith("```"):
                clean_code = clean_code[:-3].strip()
            previous_attempt = f"""

Here is a previous implementation you might want to improve:

```python
{clean_code}
```
"""

        return f"""{prompt}
{previous_attempt}
{value_ctx}

Write your `magic_denoise` function."""

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
                "mse": None,
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
        mse = outs["mse"]
        poisson = outs.get("poisson")
        mse_normalized = outs.get("mse_normalized")
        poisson_normalized = outs.get("poisson_normalized")
        msg = outs["msg"]

        if _is_entropic_adv(self.adv_estimator):
            current_mse = mse if mse is not None else float('inf')
            reward = 1/current_mse if (correctness > 0 and current_mse > 0) else 0.0
        else:
            reward = score

        logtree.log_text(f"Problem: {self.get_question()[:200]}...")
        logtree.log_text(f"Response: {message['content']}")
        logtree.log_text(
            f"Format Valid: {'✓' if correct_format else '✗'}, "
            f"Score: {score:.4f}, MSE: {mse}, Poisson: {poisson}, Reward: {reward:.4f}, Msg: {msg}"
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
                "mse": mse,
                "poisson": poisson,
                "mse_normalized": mse_normalized,
                "poisson_normalized": poisson_normalized,
                "performance": performance,
                "performance/best": performance,
                "initial_performance": self.initial_state.value if self.initial_state.value is not None else None,
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
            next_state = DenoisingState(
                timestep=step_idx,
                code=parsed_code,
                value=performance,
                mse=mse,
                poisson=poisson,
                parent_values=parent_values,
                observation=outs.get("stdout", ""),
            )
            self.sampler.update_states([next_state], [self.initial_state], save=False)
        elif hasattr(self.sampler, 'record_failed_rollout'):
            self.sampler.record_failed_rollout(self.initial_state)

        return step_result

    def get_reference_answer(self) -> str:
        raise NotImplementedError("Reference answer not available for denoising.")


async def safe_grade(
    given_answer: str,
    step: int,
    timeout: float = 10.0,
    num_cpus_per_task: int = 4,
    eval_timeout: int = 500,
    log_path: str = "",
    state: DenoisingState = None,
) -> dict[str, Any]:
    assert timeout > 100, "Outer timeout should be longer than real task timeout."
    logtree.log_text(f"Reached here: {given_answer}")
    loop = asyncio.get_running_loop()
    start_time = time.time()
    try:
        out = await asyncio.wait_for(
            loop.run_in_executor(
                SAFE_GRADE_EXECUTOR,
                partial(
                    verify_denoising,
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
        return {"score": 0.0, "msg": "Timeout grading", "correctness": 0.0, "performance": 0.0, "mse": None, "stdout": ""}
    except Exception as e:
        logger.warning(f"Exception while grading: {e}")
        return {"score": 0.0, "msg": f"Error grading: {e}", "correctness": 0.0, "performance": 0.0, "mse": None, "stdout": ""}
    return out


class DenoisingDataset(RLDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        sampler: StateSampler,
        split: Literal["train", "test"] = "train",
        seed: int = 0,
        num_cpus_per_task: int = 4,
        eval_timeout: int = 500,
        dataset_timeout: int = 300,
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
        self.sampler.flush(step)

    def _make_env_group_builder(
        self, initial_state: DenoisingState, group_size: int
    ) -> ProblemGroupBuilder:
        return ProblemGroupBuilder(
            env_thunk=partial(
                DenoisingEnv,
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
class DenoisingDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    seed: int = 0
    num_cpus_per_task: int = 4
    eval_timeout: int = 600
    dataset_timeout: int = 300
    sampler_type: str = "per"
    initial_exp_type: str = "none"
    log_path: str = ""
    problem_idx: str = "v1"
    adv_estimator: str | None = None

    async def __call__(self) -> tuple[DenoisingDataset, DenoisingDataset]:
        if not self.log_path:
            raise ValueError("log_path is required for denoising dataset")
        
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        
        sampler = get_or_create_sampler_with_default(
            self.sampler_type, self.log_path, "denoising", budget_s=self.dataset_timeout, initial_exp_type=self.initial_exp_type, batch_size=self.batch_size, problem_idx=self.problem_idx, group_size=self.group_size,
        )
        
        datasets = [
            DenoisingDataset(
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
    "denoising": DenoisingDatasetBuilder
}


def get_denoising_dataset_builder(
    dataset_path: str,
    dataset_name: str,
    batch_size: int,
    model_name_for_tokenizer: str,
    renderer_name: str,
    group_size: int,
    seed: int = 0,
    num_cpus_per_task: int = 4,
    eval_timeout: int = 600,
    dataset_timeout: int = 300,
    sampler_type: str = "per",
    initial_exp_type: str = "none",
    log_path: str = "",
    problem_idx: str = "v1",
    adv_estimator: str | None = None,
) -> RLDatasetBuilder:
    assert dataset_timeout <= eval_timeout, f"Should not set eval timeout less than dataset timeout: dataset_timeout={dataset_timeout}, eval_timeout={eval_timeout}"
    
    if dataset_name not in DATASET_BUILDER_MAP:
        raise ValueError(f"Unknown denoising dataset: {dataset_name}. Available: {list(DATASET_BUILDER_MAP.keys())}")

    if sampler_type not in SAMPLER_TYPES:
        raise ValueError(f"Unknown sampler_type: {sampler_type}. Supported: {SAMPLER_TYPES}")
    if initial_exp_type not in INITIAL_EXP_TYPES:
        raise ValueError(f"Unknown initial_exp_type: {initial_exp_type}. Supported: {INITIAL_EXP_TYPES}")
    
    if not log_path:
        raise ValueError("log_path is required for denoising dataset")

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
