import asyncio
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from functools import partial
from types import SimpleNamespace
from typing import Any, Callable, Dict, Literal, Sequence, Tuple, TypeVar
import re
import chz
from datasets import load_dataset
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import Action, EnvGroupBuilder, RLDataset, RLDatasetBuilder, StepResult
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logtree

from tasks.alphaevolve_ac.task_ae import ACInequalitiesTaskAE
from tasks.alphaevolve_ac2.task import AC2InequalitiesTask
from tinker_cookbook.recipes.ttt.state import (
    InequalitiesState,
    State,
)
from tinker_cookbook.recipes.ttt.sampler import StateSampler, get_or_create_sampler_with_default, SAMPLER_TYPES, INITIAL_EXP_TYPES
from tasks.alphaevolve_ac2.prompt import ae_verifier_program

"""
Autocorrelation Inequality (AC) environment for RL training.
"""


FINAL_MARKER = "<|end|><|start|>assistant<|channel|>final<|message|>"
CODE_BLOCK_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL)

def check_final_marker_and_code(text: str) -> Tuple[bool, bool]:
    """Returns (has_final_marker, has_valid_code_after_marker)."""
    has_marker = FINAL_MARKER in text
    if not has_marker:
        return False, False
    after_marker = text.split(FINAL_MARKER, 1)[1]
    match = CODE_BLOCK_RE.search(after_marker)
    has_valid_code = match is not None and len(match.group(1).strip()) > 0
    return True, has_valid_code

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


AC1_LITERATURE = r"""A previous state of the art used the following approach. You can use it as inspiration, but you are not required to use it, and you are encouraged to explore.
```latex
Starting from a nonnegative step function $f=(a_0,\dots,a_{n-1})$ normalized so that $\sum_j a_j=\sqrt{2n}$, set $M=\|f*f\|_\infty$. Next compute $g_0=(b_0,\dots,b_{n-1})$ by solving a linear program, i.e.\ maximizing $\sum_j b_j$ subject to $b_j\ge0$ and $\|f*g_0\|_\infty\le M$; as is standard, the optimum is attained at an extreme point determined by an active set of binding inequalities, here corresponding to important constraints where the convolution bound $(f*g_0)(x)\le M$ is tight and limiting. Rescale $g_0$ to match the normalization, $g=\frac{\sqrt{2n}}{\sum_j b_j}g_0$, and update $f\leftarrow (1-t)f+t g$ for a small $t>0$. Repeating this step produces a sequence with nonincreasing $\|f*f\|_\infty$, and the iteration is continued until it stabilizes.
```"""

AC1_EVAL_FUNCTION = '''```python
import numpy as np

def evaluate_sequence(sequence: list[float]) -> float:
    """
    Evaluates a sequence of coefficients with enhanced security checks.
    Returns np.inf if the input is invalid.
    """
    # --- Security Checks ---

    # Verify that the input is a list
    if not isinstance(sequence, list):
        return np.inf

    # Reject empty lists
    if not sequence:
        return np.inf

    # Check each element in the list for validity
    for x in sequence:
        # Reject boolean types (as they are a subclass of int) and
        # any other non-integer/non-float types (like strings or complex numbers).
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return np.inf

        # Reject Not-a-Number (NaN) and infinity values.
        if np.isnan(x) or np.isinf(x):
            return np.inf

    # Convert all elements to float for consistency
    sequence = [float(x) for x in sequence]

    # Protect against negative numbers
    sequence = [max(0, x) for x in sequence]

    # Protect against numbers that are too large
    sequence = [min(1000.0, x) for x in sequence]

    n = len(sequence)
    b_sequence = np.convolve(sequence, sequence)
    max_b = max(b_sequence)
    sum_a = np.sum(sequence)

    # Protect against the case where the sum is too close to zero
    if sum_a < 0.01:
        return np.inf

    return float(2 * n * max_b / (sum_a**2))
```'''

AC1_IMPROVEMENT_TEMPLATE = f'''Act as an expert software developer and inequality specialist specializing in creating step functions with certain properties.

Your task is to generate the sequence of non-negative heights of a step function, that minimizes the following evaluation function:

{AC1_EVAL_FUNCTION}

{AC1_LITERATURE}

Your task is to write a search function that searches for the best sequence of coefficients. Your function will have <<<BUDGET_S>>> seconds to run, and after that it has to have returned the best sequence it found. If after <<<BUDGET_S>>> seconds it has not returned anything, it will be terminated with negative infinity points. All numbers in your sequence have to be positive or zero. Larger sequences with 1000s of items often have better attack surface, but too large sequences with 100s of thousands of items may be too slow to search.

You may code up any search method you want, and you are allowed to call the evaluate_sequence() function as many times as you want. You have access to it, you don't need to code up the evaluate_sequence() function.

Here is the last code we ran:
<<<LAST_CODE>>>

<<<VALUE_CONTEXT>>>

You may want to start your search from one of the constructions we have found so far, which you can access through the 'height_sequence_1' global variable. 
However, you are encouraged to explore solutions that use other starting points to prevent getting stuck in a local minimum.

Reason about how you could further improve this construction.
Ideally, try to do something different than the above algorithm. Could be using different algorithmic ideas, adjusting your heuristics, adjusting / sweeping your hyperparemeters, etc. 
Unless you make a meaningful improvement, you will not be rewarded.

Rules:
- You must define the `propose_candidate` function as this is what will be invoked.
- You can use scientific libraries like scipy, numpy, cvxpy[CBC,CVXOPT,GLOP,GLPK,GUROBI,MOSEK,PDLP,SCIP,XPRESS,ECOS], math.
- You can use up to <<<CPUS>>> CPUs.
- Make all helper functions top level and have no closures from function nesting. Don't use any lambda functions.
- No filesystem or network IO.
- Do not import evaluate_sequence yourself. Assume it will already be imported and can be directly invoked.
- **Print statements**: Use `print()` to log progress, intermediate bounds, timing info, etc. Your output will be shown back to you.
- Include a short docstring at the top summarizing your algorithm.

Make sure to think and return the final program between ```python and ```.'''


# https://arxiv.org/pdf/2506.16750
AC2_LITERATURE = r"""A previous state of the art used the following approach. You can use it as inspiration, but you are not required to use it, and you are encoraged to explore.
```latex
Their procedure is a coarse-to-fine optimization of the score. It starts with a stochastic global search that repeatedly perturbs the current best candidate and keeps the perturbation whenever it improves (Q), with the perturbation scale gradually reduced over time. Once a good basin is found, they switch to a deterministic local improvement step, performing projected gradient ascent (move in the gradient direction and project back to the feasible region). To reach higher resolution, they lift a good low-resolution solution to a higher-dimensional one by simply repeating its entries and then rerun the local refinement. Iterating this explore–refine–upscale cycle yields their final high-resolution maximizer and the improved lower bound.
```"""

AC2_IMPROVEMENT_TEMPLATE = f'''Act as an expert software developer and inequality specialist specializing in creating step functions with certain properties.

Your task is to generate the sequence of non-negative heights of a step functions, that maximizes the following evaluation function:

```python
{ae_verifier_program}
```

{AC2_LITERATURE}
Your task is to write a search function, construct_function(), that searches for the best sequence of coefficients. Your function will have <<<BUDGET_S>>> seconds to run, and after that it has to have returned the best sequence it found. If after <<<BUDGET_S>>> seconds it has not returned anything, it will be terminated with negative infinity points. All numbers in your sequence have to be positive or zero. Larger sequences with 1000s of items often have better attack surface, but too large sequences with 100s of thousands of items may be too slow to search.

You may code up any search method you want, and you are allowed to call the evaluate_sequence() function as many times as you want. You have access to it, you don't need to code up the evaluate_sequence() function.

Here is the last code we ran:
<<<LAST_CODE>>>

<<<VALUE_CONTEXT>>>

You may want to start your search from one of the constructions we have found so far, which you can access through the 'height_sequence_1' global variable. 
However, you are encouraged to explore solutions that use other starting points to prevent getting stuck in a local minimum.

Reason about how you could further improve this construction.
Ideally, try to do something different than the above algorithm. Could be using different algorithmic ideas, adjusting your heuristics, adjusting / sweeping your hyperparemeters, etc. 
Unless you make a meaningful improvement, you will not be rewarded, if you are stuck you should think about how to get unstuck.

Rules:
- You must define the `construct_function` function as this is what will be invoked.
- You can use scientific libraries like scipy, numpy, cvxpy[CBC,CVXOPT,GLOP,GLPK,GUROBI,MOSEK,PDLP,SCIP,XPRESS,ECOS], math.
- You can use up to <<<CPUS>>> CPUs.
- Make all helper functions top level and have no closures from function nesting. Don't use any lambda functions.
- No filesystem or network IO.
- Do not import evaluate_sequence yourself. Assume it will already be imported and can be directly invoked. Do not import height_sequence_1 yourself; it will already be available.
- **Print statements**: Use `print()` to log progress, intermediate bounds, timing info, etc. Your output will be shown back to you.
- Include a short docstring at the top summarizing your algorithm.

Make sure to think and return the final program between ```python and ```.
'''


# Tune this based on your node size / num_cpus_per_task
SAFE_GRADE_MAX_WORKERS = 4096
SAFE_GRADE_EXECUTOR = ThreadPoolExecutor(max_workers=SAFE_GRADE_MAX_WORKERS)


def dict_to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_ns(v) for k, v in d.items()})
    return d
    
default_config = {
    "ttt_rm": {
        "num_cpus_per_task": 2,
        "rew_type": "scaled_reciprocal_cf",
        "fail_score": 0.0,
        "eval_timeout": 1100,
        "worst_perf_log": -10000,
        "n_item": 26,
        "sweep_hyperparams": False,
        "max_hyperparam_combos": 16,
    }
}

def verify_ac(
    generation: str,
    step: int,
    sweep_hyperparams: bool = False,
    max_hyperparam_combos: int = 16,
    num_cpus_per_task: int = 1,
    env_name: str = "ac1",
    problem_idx: str = "improvement",
    eval_timeout: int = 300,
    log_path: str = "",
    state: InequalitiesState = None,
) -> dict:
    config = default_config.copy()
    config["ttt_rm"] = default_config["ttt_rm"].copy()
    config["ttt_rm"]["sweep_hyperparams"] = sweep_hyperparams
    config["ttt_rm"]["max_hyperparam_combos"] = max_hyperparam_combos
    config["ttt_rm"]["num_cpus_per_task"] = num_cpus_per_task
    config["ttt_rm"]["eval_timeout"] = eval_timeout
    config_ns = dict_to_ns(config)

    logtree.log_text(f"Starting gen, {config}")

    if env_name == "ac1":
        task = ACInequalitiesTaskAE(config_ns, log_path)
    elif env_name == "ac2":
        logtree.log_text(f"Using AC2")
        config["ttt_rm"]["rew_type"] = "linear"
        config_ns = dict_to_ns(config)
        task = AC2InequalitiesTask(config_ns, log_path)
    else:
        raise ValueError(f"Received an unexpected env_name for ACInequalities env: {env_name}")

    out = task.compute_score(generation, step=step, state=state)

    return {
        "score": out["score"],
        "msg": out["msg"],
        "correctness": out["correctness"],
        "performance": out["performance"],  # upper bound (lower is better)
        "result_construction": out.get("result_construction", None),
        "stdout": out.get("stdout", ""),
    }


logger = logging.getLogger(__name__)
T = TypeVar("T")


def _is_entropic_adv(adv_estimator: str | None) -> bool:
    return adv_estimator in ("entropic", "entropic_adaptive_beta")


class AutoCorrInequalityEnv(ProblemEnv):
    def __init__(
        self,
        renderer: renderers.Renderer,
        initial_state: State,
        sampler: StateSampler,
        timeout: float = 6000.0,
        sweep_hyperparams: bool = False,
        max_hyperparam_combos: int = 16,
        num_cpus_per_task: int = 1,
        env_name: str = "ac1",
        problem_idx: str = "improvement",
        eval_timeout: int = 300,
        log_path: str = "",
        convo_prefix=None,
        budget_s: int = 1000,
        adv_estimator: str | None = None,
    ):
        super().__init__(renderer, convo_prefix=convo_prefix)
        
        if initial_state is None:
            raise ValueError("initial_state is required and cannot be None")
        if sampler is None:
            raise ValueError("sampler is required and cannot be None")
        
        self.timeout = timeout
        self.sweep_hyperparams = sweep_hyperparams
        self.max_hyperparam_combos = max_hyperparam_combos
        self.num_cpus_per_task = num_cpus_per_task
        self.env_name = env_name
        self.problem_idx = problem_idx
        self.eval_timeout = eval_timeout
        self.log_path = log_path
        self.initial_state = initial_state
        self.sampler = sampler
        self.budget_s = budget_s
        self.adv_estimator = adv_estimator

        self.state = initial_state

    def get_question(self) -> str:
        """Build prompt from template, injecting previous code from state."""
        return self._get_improvement_prompt(self.initial_state)

    def _get_improvement_prompt(self, state: InequalitiesState) -> str:
        """Build contextual improvement prompt with before/after values."""
        # state_only mode: show construction but not code (forces model to write fresh code)
        hide_code = "state_only" in self.problem_idx
        has_code = state.code and state.code.strip() and not hide_code
        
        # Value context: show before/after if we have parent values
        # state.value is -upper_bound (higher=better for RL), so negate to get upper_bound (lower=better)
        if state.parent_values and state.value is not None:
            if self.env_name == "ac1":
                before_bound = -state.parent_values[0]
                after_bound = -state.value
                value_ctx = f"\nHere are the upper bounds before and after running the code above (lower is better): {before_bound:.10f} -> {after_bound:.10f}"
                value_ctx += f"\nOur target is to make the upper bound tighter, just as a reference, lower it to at least 1.5030. Further improvements will also be generously rewarded."
            else:
                before_bound = state.parent_values[0]
                after_bound = state.value
                value_ctx = f"\nHere are the lower bounds before and after running the code above (higher is better): {before_bound:.10f} -> {after_bound:.10f}"
                value_ctx += f"\nOur target is to make the lower bound tighter, just as a reference, close to at least 0.97. Further improvements will also be generously rewarded."
        elif state.value is not None:
            if self.env_name == "ac1":
                value_ctx = f"\nCurrent upper bound (lower is better): {-state.value:.10f}"
                value_ctx += f"\nOur target is to make the upper bound tighter, just as a reference, close to at least 1.5030. Further improvements will also be generously rewarded."
            else:
                value_ctx = f"\nCurrent lower bound (higher is better): {state.value:.10f}"
                value_ctx += f"\nOur target is to make the lower bound tighter, just as a reference, close to at least 0.97. Further improvements will also be generously rewarded."
        else:
            value_ctx = ""
        
        if state.construction:
            value_ctx += f"\nLength of the construction: {len(state.construction)}"
        
        # Show previous stdout if available
        if state.observation and state.observation.strip():
            stdout = state.observation.strip()
            if len(stdout) > 500:
                stdout = "\n\n\t\t ...(TRUNCATED)...\n" + stdout[-500:]
            value_ctx += f"\n\n--- Previous Program Output ---\n{stdout}\n--- End Output ---"
        
        prompt = AC1_IMPROVEMENT_TEMPLATE if self.env_name == "ac1" else AC2_IMPROVEMENT_TEMPLATE
        
        if has_code:
            prompt = prompt.replace("<<<LAST_CODE>>>", state.code)
        else:
            prompt = prompt.replace("Here is the last code we ran:\n<<<LAST_CODE>>>\n\n", "")
            prompt = prompt.replace("You are iteratively optimizing constructions.", "")
            prompt = prompt.replace("Reason about how you could further improve this construction.", 
                                    "Write code to optimize this construction.")
        
        prompt = prompt.replace("<<<VALUE_CONTEXT>>>", value_ctx)
        prompt = prompt.replace("<<<BUDGET_S>>>", str(self.budget_s))
        prompt = prompt.replace("<<<CPUS>>>", str(self.num_cpus_per_task))
        if isinstance(self.renderer, renderers.Qwen3Renderer):
            prompt = prompt.replace("Make sure to think", "Make sure to /think")
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
                "result_construction": None,
            }
        # Async grading: runs verify_ac1 in a thread with timeout
        return await safe_grade(
            parsed_code, 
            step,
            self.timeout, 
            self.sweep_hyperparams, 
            self.max_hyperparam_combos, 
            self.num_cpus_per_task,
            self.env_name,
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
        score = outs["score"]
        correctness = outs["correctness"]
        performance = outs["performance"]  # upper bound (lower is better)
        msg = outs["msg"]
        result_construction = outs.get("result_construction", None)

        if self.env_name == "ac1":
            # AC1: lower upper_bound is better, target=1.5030
            # current_bound = -performance (actual upper bound)
            # reward = goal/current_bound - 1 (inverse because lower is better)
            # When current_bound = goal: reward = 0
            # When current_bound < goal: reward > 0 (better)
            # When current_bound > goal: reward < 0 (worse)
            current_bound = -performance  # actual upper bound, e.g., 1.510
            # Reciprocal, often catches 3 digits precision
            reward = 1/current_bound if (correctness > 0) else 0.0
        elif self.env_name == "ac2":
            # AC2: higher lower_bound is better, target=1.0
            # current_bound = performance (actual lower bound)
            # reward = current_bound/goal - 1 (direct because higher is better)
            # When current_bound = goal: reward = 0
            # When current_bound > goal: reward > 0 (better)
            # When current_bound < goal: reward < 0 (worse)
            current_bound = performance  # actual lower bound
            # Catch 3 digits precision
            reward = current_bound if correctness > 0 else 0.0
        else:
            reward = score


        # For AC1: performance = -upper_bound, so best_bound = -performance = actual upper bound
        # For AC2: performance = lower_bound, so best_bound = performance
        # Only compute best_bound for correct submissions to avoid polluting aggregates
        best_bound = (-performance if self.env_name == "ac1" else performance) if correctness > 0 else None
        
        # Check if response has final marker and valid code after it
        has_final_marker, has_valid_code = check_final_marker_and_code(response)
        
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
                "performance": best_bound,  # actual bound value, None for incorrect submissions
                "parsed_final_code": float(has_final_marker and has_valid_code),
                # AC1: upper_bound (lower is better, use min), AC2: lower_bound (higher is better, use max)
                "performance/best": best_bound,
                "initial_performance": -self.initial_state.value if self.env_name == "ac1" and self.initial_state.value is not None else self.initial_state.value,
                "msg": msg,
                "predicted_grid": None,
                "prompt": self.get_question(),
                "response": message['content'],
                "ref": msg,
            },
        )

        # Update sampler with new state if we have valid result (defer save until flush)
        if correctness > 0 and result_construction is not None:
            next_state = InequalitiesState(
                timestep=step_idx,
                construction=result_construction,
                code=parsed_code,
                value=performance,  # performance = -upper_bound, so higher = better
                observation=outs.get("stdout", ""),
            )
            self.sampler.update_states([next_state], [self.initial_state], save=False)
        elif hasattr(self.sampler, 'record_failed_rollout'):
            # Record that we tried this parent but got no valid child (for PUCT visit counts)
            self.sampler.record_failed_rollout(self.initial_state)

        return step_result

    def get_reference_answer(self) -> str:
        """Return the reference answer for logging purposes."""
        raise NotImplementedError("Reference answer not available for AC inequalities.")


async def safe_grade(
    given_answer: str,
    step: int,
    timeout: float = 10.0,
    sweep_hyperparams: bool = False,
    max_hyperparam_combos: int = 16,
    num_cpus_per_task: int = 1,
    env_name: str = "ac1",
    problem_idx: str = "improvement",
    eval_timeout: int = 300,
    log_path: str = "",
    state: InequalitiesState = None,
) -> dict[str, Any]:
    """Async grader: runs verify_ac1 in a background thread with asyncio timeout."""
    assert timeout > 300, "Outer timeout should be longer than real task timeout."
    logtree.log_text(f"Reached here: {given_answer}")
    loop = asyncio.get_running_loop()
    start_time = time.time()
    try:
        out = await asyncio.wait_for(
            loop.run_in_executor(
                SAFE_GRADE_EXECUTOR,
                partial(
                    verify_ac, 
                    given_answer, 
                    step,
                    sweep_hyperparams, 
                    max_hyperparam_combos, 
                    num_cpus_per_task, 
                    env_name,
                    problem_idx,
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
        return {"score": 0.0, "msg": "Timeout grading", "correctness": 0.0, "performance": 0.0, "result_construction": None, "stdout": ""}
    except Exception as e:
        logger.warning(f"Exception while grading: {e}")
        return {"score": 0.0, "msg": f"Error grading: {e}", "correctness": 0.0, "performance": 0.0, "result_construction": None, "stdout": ""}
    return out


class AutoCorrInequalityDataset(RLDataset):
    def __init__(
        self,
        env_name: str,
        problem_idx: str,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        sampler: StateSampler,
        split: Literal["train", "test"] = "train",
        seed: int = 0,
        sweep_hyperparams: bool = False,
        max_hyperparam_combos: int = 16,
        num_cpus_per_task: int = 1,
        eval_timeout: int = 300,
        dataset_timeout: int = 300,
        log_path: str = "",
        adv_estimator: str | None = None,
    ):
        self.split = split
        self.batch_size = batch_size
        self.group_size = group_size if split == "train" else 1
        self.renderer = renderer
        self.sweep_hyperparams = sweep_hyperparams
        self.max_hyperparam_combos = max_hyperparam_combos
        self.num_cpus_per_task = num_cpus_per_task
        self.env_name = env_name
        self.problem_idx = problem_idx
        self.eval_timeout = eval_timeout
        self.dataset_timeout = dataset_timeout
        self.log_path = log_path
        self.sampler = sampler
        self.adv_estimator = adv_estimator
    
    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        """Get batch of env group builders, each with a sampled state."""
        states = self.sampler.sample_states(self.batch_size)
        return [self._make_env_group_builder(state, self.group_size) for state in states]

    def flush(self, step: int | None = None):
        """Flush sampler state to disk. Call after batch completes."""
        self.sampler.flush(step)

    def __len__(self) -> int:
        # With sampler, we can run indefinitely. Set this via num_epochs.
        return 1

    def _make_env_group_builder(
        self, initial_state: State, group_size: int
    ) -> ProblemGroupBuilder:
        return ProblemGroupBuilder(
            env_thunk=partial(
                AutoCorrInequalityEnv,
                self.renderer,
                initial_state=initial_state,
                sampler=self.sampler,
                sweep_hyperparams=self.sweep_hyperparams,
                max_hyperparam_combos=self.max_hyperparam_combos,
                num_cpus_per_task=self.num_cpus_per_task,
                env_name=self.env_name,
                problem_idx=self.problem_idx,
                eval_timeout=self.eval_timeout,
                log_path=self.log_path,
                convo_prefix=None,
                budget_s=self.dataset_timeout,
                adv_estimator=self.adv_estimator,
            ),
            num_envs=group_size,
        )


@chz.chz
class AutoCorrInequalityDatasetBuilder(RLDatasetBuilder):
    env_name: str
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    seed: int = 0
    problem_idx: str = "improvement"
    sweep_hyperparams: bool = False
    max_hyperparam_combos: int = 16
    num_cpus_per_task: int = 1
    eval_timeout: int = 300
    dataset_timeout: int = 300
    sampler_type: str = "greedy"
    initial_exp_type: str = "random"  # "best_available", "none", "random"
    log_path: str = ""
    adv_estimator: str | None = None


    async def __call__(self) -> tuple[AutoCorrInequalityDataset, AutoCorrInequalityDataset]:
        if not self.log_path:
            raise ValueError("log_path is required for AC dataset")
        
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        
        sampler = get_or_create_sampler_with_default(
            self.sampler_type, self.log_path, self.env_name, budget_s=self.dataset_timeout, initial_exp_type=self.initial_exp_type, batch_size=self.batch_size, problem_idx=self.problem_idx, group_size=self.group_size,
        )
        
        datasets = [
            AutoCorrInequalityDataset(
                env_name=self.env_name,
                problem_idx=self.problem_idx,
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                sampler=sampler,
                split=split,
                seed=self.seed,
                sweep_hyperparams=self.sweep_hyperparams,
                max_hyperparam_combos=self.max_hyperparam_combos,
                num_cpus_per_task=self.num_cpus_per_task,
                eval_timeout=self.eval_timeout,
                dataset_timeout=self.dataset_timeout,
                log_path=self.log_path,
                adv_estimator=self.adv_estimator,
            )
            for split in ("train", "test")
        ]
        return (datasets[0], datasets[1])


def get_ac_dataset_builder(
    dataset_path: str,
    dataset_name: str,
    batch_size: int,
    model_name_for_tokenizer: str,
    renderer_name: str,
    group_size: int,
    problem_idx: str = "improvement",
    seed: int = 0,
    sweep_hyperparams: bool = False,
    max_hyperparam_combos: int = 16,
    num_cpus_per_task: int = 1,
    eval_timeout: int = 300,
    dataset_timeout: int = 300,
    sampler_type: str = "greedy",
    initial_exp_type: str = "random",
    log_path: str = "",
    adv_estimator: str | None = None,
) -> RLDatasetBuilder:
    """
    Unified function to get any ac dataset builder.
    Args:
        dataset_name: One of "ac1" "ac2"
        batch_size: Number of groups per batch
        model_name_for_tokenizer: Model name for tokenizer
        renderer_name: Name of the renderer to use
        group_size: Number of environments per group
        problem_idx: AC variant
        seed: Random seed for data shuffling (default: 0)
        sweep_hyperparams: Whether to sweep over hyperparameters (default: False)
        max_hyperparam_combos: Maximum number of hyperparameter combinations (default: 16)
        sampler_type: Type of experience sampler
        initial_exp_type: Type of initial experience: "best_available", "none", "random"
    Returns:
        The appropriate dataset builder instance
    """

    # General catches to make sure we don't launch mismatched dataset and eval timeouts
    assert dataset_timeout <= eval_timeout, f"Should not set eval timeout less than dataset timeout: dataset_timeout={dataset_timeout}, eval_timeout={eval_timeout}"
    assert dataset_timeout + 200 >= eval_timeout, f"Found eval timeout unexpectedly much higher than dataset timeout: dataset_timeout={dataset_timeout}, eval_timeout={eval_timeout}"
   
    if sampler_type not in SAMPLER_TYPES:
        raise ValueError(f"Unknown sampler_type: {sampler_type}. Supported: {SAMPLER_TYPES}")
    if initial_exp_type not in INITIAL_EXP_TYPES:
        raise ValueError(f"Unknown initial_exp_type: {initial_exp_type}. Supported: {INITIAL_EXP_TYPES}")
    
    if not log_path:
        raise ValueError("log_path is required for AC dataset")

    # Run the init of task to create ray namespace actors
    # Removes race condition error when first two tasks are run at the same time
    config_ns = dict_to_ns(default_config)
    ACInequalitiesTaskAE(config_ns, log_path)

    return AutoCorrInequalityDatasetBuilder(
        env_name=dataset_name,
        batch_size=batch_size,
        model_name_for_tokenizer=model_name_for_tokenizer,
        renderer_name=renderer_name,
        group_size=group_size,
        seed=seed,
        problem_idx=problem_idx,
        sweep_hyperparams=sweep_hyperparams,
        max_hyperparam_combos=max_hyperparam_combos,
        num_cpus_per_task=num_cpus_per_task,
        eval_timeout=eval_timeout,
        dataset_timeout=dataset_timeout,
        sampler_type=sampler_type,
        initial_exp_type=initial_exp_type,
        log_path=log_path,
        adv_estimator=adv_estimator,
    )
