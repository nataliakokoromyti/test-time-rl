NVFP4_GROUP_GEMM_IMPROVEMENT_TEMPLATE_V1 = '''You are an expert Triton engineer tasked with implementing a block scaled group matrix-matrix multiplication kernel optimized for NVIDIA B200.

The kernel operates over grouped matrix shapes and low-precision formats. You will be given a tuple:
```
(abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes)
```

Where:
- abc_tensors: list of (a, b, c)
  - a: torch.Tensor[float4e2m1fn_x2] shape [M, K // 2, L]
  - b: torch.Tensor[float4e2m1fn_x2] shape [N, K // 2, L]
  - c: torch.Tensor[float16] shape [M, N, L]
- sfasfb_tensors: list of (sfa, sfb)
  - sfa: torch.Tensor[float8_e4m3fnuz] shape [M, K // 16, L]
  - sfb: torch.Tensor[float8_e4m3fnuz] shape [N, K // 16, L]
- sfasfb_reordered_tensors: list of (sfa_reordered, sfb_reordered)
  - sfa_reordered: torch.Tensor[float8_e4m3fnuz] shape [32, 4, rest_m, 4, rest_k, L]
  - sfb_reordered: torch.Tensor[float8_e4m3fnuz] shape [32, 4, rest_n, 4, rest_k, L]
- problem_sizes: list of (M, N, K, L)

Each group has its own tensors and sizes. L is always 1. M and N are divisible by mma_tiler_mn, and K is divisible by 256.

Your function must be named `custom_kernel` and return a list of output tensors.

Here is a PyTorch reference implementation of NVFP4 block-scaled group GEMM:
```python
import torch
from task import input_t, output_t

# Scaling factor vector size
sf_vec_size = 16

# Helper function for ceiling division
def ceil_div(a, b):
    return (a + b - 1) // b

# Helper function to convert scale factor tensor to blocked format
def to_blocked(input_matrix):
    rows, cols = input_matrix.shape

    # Please ensure rows and cols are multiples of 128 and 4 respectively
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    # Pad the input matrix if necessary
    if padded_rows != rows or padded_cols != cols:
        padded = torch.nn.functional.pad(
            input_matrix,
            (0, padded_cols - cols, 0, padded_rows - rows),
            mode="constant",
            value=0,
        )
    else:
        padded = input_matrix
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)

    return rearranged.flatten()


def ref_kernel(
    data: input_t,
) -> output_t:
    """
    PyTorch reference implementation of NVFP4 block-scaled group GEMM.
    """
    abc_tensors, sfasfb_tensors, _, problem_sizes = data

    result_tensors = []
    for i, (
        (a_ref, b_ref, c_ref),
        (sfa_ref, sfb_ref),
        (m, n, k, l),
    ) in enumerate(
        zip(
            abc_tensors,
            sfasfb_tensors,
            problem_sizes,
        )
    ):
        for l_idx in range(l):
            # Convert the scale factor tensor to blocked format
            scale_a = to_blocked(sfa_ref[:, :, l_idx])
            scale_b = to_blocked(sfb_ref[:, :, l_idx])
            # (m, k) @ (n, k).T -> (m, n)
            res = torch._scaled_mm(
                a_ref[:, :, l_idx].view(torch.float4_e2m1fn_x2),
                b_ref[:, :, l_idx].transpose(0, 1).view(torch.float4_e2m1fn_x2),
                scale_a.cuda(),
                scale_b.cuda(),
                bias=None,
                out_dtype=torch.float16,
            )
            c_ref[:, :, l_idx] = res
        result_tensors.append((c_ref))
    return result_tensors
```

Reference template:
```python
from task import input_t, output_t

def custom_kernel(data: input_t) -> output_t:
    abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes = data
    result_tensors = []
    for i, ((a, b, c), (sfa_reordered, sfb_reordered), (m, n, k, l)) in enumerate(
        zip(abc_tensors, sfasfb_reordered_tensors, problem_sizes)
    ):
        # add your implementation here
        result_tensors.append(c)

    return result_tensors
```

Test cases for correctness and runtime (optimize runtime for these):
  - tests/benchmarks are defined in the task definition and evaluated on B200.

Here is the last code we ran:
```python
<<<LAST_CODE>>>
```

<<<VALUE_CONTEXT>>>

Token budget rules (verbatim from the paper):
- context_window = 32768
- max_tokens = context_window - prompt_length
- prompt + thinking tokens <= 26000
- remaining tokens are reserved for the final response
- teacher forcing message (emit exactly when you hit the thinking limit):
  "... okay, I am out of thinking tokens. I need to send my final message now."

Rules:
- Define all of your code in one final ```python ``` block.
- The entrypoint to your code must be named `custom_kernel`.
- Optimize for runtime on NVIDIA B200.
- You may use Triton/CUDA/PyTorch as needed.
- Include a short docstring at the top summarizing your approach.
'''
