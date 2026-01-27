from types import SimpleNamespace
from typing import Any

from tinker_cookbook.utils import logtree

from tasks.denoising.task import DenoisingTask, MAGIC_FUNC, EVALUATE_MSE_FUNC, EVALUATE_POISSON_FUNC
from tasks.denoising.verifier import BASELINES
from tasks.denoising.prompt import SYSTEM_PROMPT
from tinker_cookbook.recipes.ttt.state import DenoisingState
from tinker_cookbook.recipes.ttt.env_ttt import BaseTTTEnv
from tinker_cookbook.recipes.ttt.dataset_builder import DatasetConfig


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


def _is_entropic_adv(adv_estimator: str | None) -> bool:
    return adv_estimator in ("entropic", "entropic_adaptive_beta")


class DenoisingEnv(BaseTTTEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    def _verify_code(
        self,
        generation: str,
        step: int,
        num_cpus_per_task: int = 4,
        eval_timeout: int = 500,
        log_path: str = "",
        state: DenoisingState = None,
        **kwargs
    ) -> dict[str, Any]:
        return verify_denoising(generation, step, num_cpus_per_task, eval_timeout, log_path, state)

    def _get_verify_kwargs(self) -> dict[str, Any]:
        return {
            "num_cpus_per_task": self.num_cpus_per_task,
            "eval_timeout": self.eval_timeout,
            "log_path": self.log_path,
            "state": self.state,
        }

    def _get_timeout_response(self) -> dict[str, Any]:
        return {
            "score": 0.0,
            "msg": "Timeout grading",
            "correctness": 0.0,
            "performance": 0.0,
            "mse": None,
            "stdout": "",
        }

    def _get_error_response(self, error_msg: str) -> dict[str, Any]:
        return {
            "score": 0.0,
            "msg": f"Error grading: {error_msg}",
            "correctness": 0.0,
            "performance": 0.0,
            "mse": None,
            "stdout": "",
        }

    def _compute_reward(self, outs: dict[str, Any], correctness: float) -> float:
        mse = outs.get("mse")
        if _is_entropic_adv(self.adv_estimator):
            current_mse = mse if mse is not None else float('inf')
            return 1/current_mse if (correctness > 0 and current_mse > 0) else 0.0
        else:
            return outs["score"]

    def _create_next_state(
        self,
        step_idx: int,
        parsed_code: str,
        outs: dict[str, Any],
    ) -> DenoisingState:
        performance = outs.get("performance")
        if performance is None:
            return None
        parent_state = self.initial_state
        parent_values = [parent_state.value] + parent_state.parent_values if parent_state.value is not None else []
        return DenoisingState(
            timestep=step_idx,
            code=parsed_code,
            value=performance,
            mse=outs.get("mse"),
            poisson=outs.get("poisson"),
            parent_values=parent_values,
            observation=outs.get("stdout", ""),
        )

    def _build_metrics(
        self,
        outs: dict[str, Any],
        correct_format: bool,
        message: dict,
        parsed_code: str,
    ) -> dict[str, Any]:
        return {
            "format": correct_format,
            "score": outs["score"],
            "correctness": outs["correctness"],
            "correct": outs["correctness"],
            "mse": outs.get("mse"),
            "poisson": outs.get("poisson"),
            "mse_normalized": outs.get("mse_normalized"),
            "poisson_normalized": outs.get("poisson_normalized"),
            "performance": outs.get("performance"),
            "performance/best": outs.get("performance"),
            "initial_performance": self.initial_state.value if self.initial_state.value is not None else None,
            "msg": outs.get("msg", ""),
            "predicted_grid": None,
            "prompt": self.get_question(),
            "response": message['content'],
            "ref": outs.get("msg", ""),
        }


