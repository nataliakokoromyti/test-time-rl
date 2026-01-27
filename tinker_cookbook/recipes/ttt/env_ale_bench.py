from typing import Any

from tinker_cookbook.utils import logtree

from tasks.ale_bench.prompt import create_prompt
from tasks.ale_bench.task import run_ale_bench_task
from tinker_cookbook.recipes.ttt.state import AleBenchState, State
from tinker_cookbook.recipes.ttt.env_ttt import BaseTTTEnv
from tinker_cookbook.recipes.ttt.dataset_builder import DatasetConfig


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


def parse_ale_bench_problem_idx(problem_idx: str) -> tuple[int, str | None]:
    if "_improvement" in problem_idx:
        problem_id = problem_idx.replace("_improvement", "")
        return problem_id, "v1"
    return problem_idx, "v1"


class AleBenchEnv(BaseTTTEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        problem_id, improvement_version = parse_ale_bench_problem_idx(self.problem_idx)
        self.problem_idx = problem_id
        self.improvement_version = improvement_version

    def _get_improvement_prompt(self, state: AleBenchState) -> str:
        """Build contextual improvement prompt with value context."""
        assert self.problem_idx in {"ahc039", "ahc058"}
        
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

    def _get_code_languages(self) -> list[str]:
        return ["cpp"]
    
    def _should_keep_code_separators(self) -> bool:
        return False  # ALE Bench doesn't keep separators

    def _verify_code(
        self,
        generation: str,
        step: int,
        num_cpus_per_task: int | None = None,
        problem_idx: str | None = None,
        log_dir: str | None = None,
        **kwargs
    ) -> dict[str, Any]:
        return verify_ale_bench(generation, step, num_cpus_per_task, problem_idx, log_dir)

    def _get_verify_kwargs(self) -> dict[str, Any]:
        return {
            "num_cpus_per_task": self.num_cpus_per_task,
            "problem_idx": self.problem_idx,
            "log_dir": self.log_path,
        }

    def _compute_reward(self, outs: dict[str, Any], correctness: float) -> float:
        return outs["score"] if correctness > 0 else 0.0

    def _create_next_state(
        self,
        step_idx: int,
        parsed_code: str,
        outs: dict[str, Any],
    ) -> AleBenchState:
        return AleBenchState(
            timestep=step_idx,
            code=parsed_code,
            value=outs["performance"],  # higher = better
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
            "runtime": outs["performance"],
            "score": outs["score"],
            "correctness": outs["correctness"],
            "correct": outs["correctness"],
            "upper_bound": outs["performance"],
            "msg": outs.get("msg", ""),
            "predicted_grid": None,
            "prompt": self.get_question(),
            "response": message['content'],
            "ref": outs.get("msg", ""),
        }
