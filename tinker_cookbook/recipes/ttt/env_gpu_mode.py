from typing import Any

from tinker_cookbook.utils import logtree

from tasks.gpu_mode.task import run_gpu_mode_task
from tasks.gpu_mode.prompt_trimul import TRIMUL_IMPROVEMENT_TEMPLATE_V0
from tasks.gpu_mode.prompt_mla_decode import MLA_DECODE_PROMPT_V1, MLA_DECODE_IMPROVEMENT_TEMPLATE_V1
from tasks.gpu_mode.prompt_nvfp4_group_gemm import NVFP4_GROUP_GEMM_IMPROVEMENT_TEMPLATE_V1

from tinker_cookbook.recipes.ttt.state import GpuModeState, State
from tinker_cookbook.recipes.ttt.env_ttt import BaseTTTEnv
from tinker_cookbook.recipes.ttt.dataset_builder import DatasetConfig


async def verify_gpu_mode(
    generation: str,
    gpu_type: str,
    task_name: str,
    score_scale: float,
    app_name: str,
) -> dict:
    logtree.log_text(f"Starting gen")

    return await run_gpu_mode_task(generation, gpu_type=gpu_type, task_name=task_name, score_scale=score_scale, app_name=app_name)


def parse_gpu_mode_problem_idx(problem_idx: str) -> tuple[str, str | None]:
    return problem_idx, "v0"


class GpuModeEnv(BaseTTTEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.problem_idx = self.config.problem_idx
        self.dataset_name = self.config.dataset_name
        match self.dataset_name:
            case "trimul":
                self.improvement_prompt = TRIMUL_IMPROVEMENT_TEMPLATE_V0
                self.target = 1000  # Human best is 1371.057μs
                self.gpu_type = "H100"
                self.task_name = "trimul"
                self.score_scale = self.config.gpu_mode_score_scale
                self.app_name = "discord-bot-runner"
            case "mla_decode_nvidia":
                self.improvement_prompt = MLA_DECODE_IMPROVEMENT_TEMPLATE_V1
                self.target = 1700  # Human best is 1787.457μs (on MI300X)
                self.gpu_type = "H200"
                self.task_name = "mla_decode_nvidia"
                self.score_scale = self.config.gpu_mode_score_scale
                self.app_name = "discord-bot-runner-mla-decode-nvidia"
            case "nvfp4_group_gemm":
                self.improvement_prompt = NVFP4_GROUP_GEMM_IMPROVEMENT_TEMPLATE_V1
                self.target = 0  # Unknown target; optimize runtime
                self.gpu_type = "B200"
                self.task_name = "nvfp4_group_gemm"
                self.score_scale = self.config.gpu_mode_score_scale
                self.app_name = "discord-bot-runner"
            case _:
                raise ValueError(f"Unknown dataset name: {self.dataset_name}")
        
        n, improvement_version = parse_gpu_mode_problem_idx(self.config.problem_idx)
        self.problem_idx = n
        self.improvement_version = improvement_version

    def get_question(self) -> str:
        state = self.initial_state
        
        # Use improvement template for iterative optimization
        return self._get_improvement_prompt(state)

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
        elif self.dataset_name == "nvfp4_group_gemm":
            if (parsed_code is None) or (parsed_code.strip() == '') or ("def custom_kernel" not in parsed_code):
                return False
            return True
        
        else:
            raise ValueError(f"Unknown dataset name: {self.dataset_name}")

    def _get_code_languages(self) -> list[str]:
        return ["python"]
    
    def _should_keep_code_separators(self) -> bool:
        return False  # GPU mode doesn't keep separators

    def _verify_code(self, generation: str, step: int, **kwargs) -> dict[str, Any]:
        """Not used - GPU mode overrides _safe_grade instead."""
        raise NotImplementedError("GPU mode uses async verify_gpu_mode directly")

    async def _safe_grade(self, given_answer: str, step: int) -> dict[str, Any]:
        """Override to handle async verify_gpu_mode directly (not via executor)."""
        import asyncio
        try:
            out = await asyncio.wait_for(
                verify_gpu_mode(
                    given_answer,
                    self.gpu_type,
                    self.task_name,
                    self.score_scale,
                    self.app_name,
                ),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            return {"score": 0.0, "msg": "Timeout grading", "correctness": 0.0, "performance": 0.0}
        except Exception as e:
            return {"score": 0.0, "msg": f"Error grading: {e}", "correctness": 0.0, "performance": 0.0}
        return out

    def _compute_reward(self, outs: dict[str, Any], correctness: float) -> float:
        return outs["score"]

    def _create_next_state(
        self,
        step_idx: int,
        parsed_code: str,
        outs: dict[str, Any],
    ) -> GpuModeState:
        # Combine msg and benchmark_details for observation
        obs_parts = [outs.get("msg", "")]
        if outs.get("benchmark_details"):
            obs_parts.append(f"\nPer-benchmark timing:\n{outs['benchmark_details']}")
        observation = "\n".join(obs_parts)
        return GpuModeState(
            timestep=step_idx,
            code=parsed_code,
            value=outs["performance"],  # higher = better, performance is -runtime
            observation=observation,
        )

    def _build_metrics(
        self,
        outs: dict[str, Any],
        correct_format: bool,
        message: dict,
        parsed_code: str,
    ) -> dict[str, Any]:
        score = outs["score"]
        correctness = outs["correctness"]
        performance = outs.get("performance")
        return {
            "format": correct_format,
            "score": score,
            "correctness": correctness,
            "correct": correctness,
            "performance": -performance if correctness > 0 else None,  # runtime in μs, None for incorrect
            "performance/best": -performance if correctness > 0 else None,  # runtime, lower is better, use min
            "initial_performance": -self.initial_state.value if self.initial_state.value is not None else None,
            "msg": outs.get("msg", ""),
            "predicted_grid": None,
            "prompt": self.get_question(),
            "response": message['content'],
            "ref": outs.get("msg", ""),
        }


