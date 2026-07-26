## Model choice matters a lot here

Base-model choice has a large effect on this example's search behavior, well beyond ordinary
capability differences. In a same-harness, same-seed-budget sweep (10 cases, `n_iters=20`,
`depth_limit=6`, after the `next_step.py` prompt-accumulation fix in
[#6](https://github.com/newexo/llm-reasoners/pull/6)):

| Model | Terminal ("Finish.") reached | Accuracy | Entity contamination |
|---|---|---|---|
| Mistral-7B-v0.1 / v0.3 | 10/10 | 60% | none |
| Llama-2-7B | 10/10 | 30% | none |
| Qwen1.5-7B | 10/10 | 30% | mild |
| Llama-3.1-8B | 9/10 | 30% | mild |
| Qwen2.5-7B-Instruct | 8/10 | 40% | none |
| Qwen2.5-7B (base) | 7/10 | 10% | severe |
| Phi-2 (2.7B) | 5/10 | 10% | none |

"Entity contamination" is the derived reasoning chain drifting onto an unrelated entity from a
few-shot demonstration instead of the actual test question's subject (e.g. a question about "Rex"
producing an answer about "Stella" or "127"). It shows up to varying degrees in Qwen (both
generations) and Llama-3.1, but not at all in Mistral (either version) or Phi-2 — so it isn't a
general small/old-base-model trait, and newer generations aren't uniformly better: Qwen2.5-base is
*worse* than Qwen1.5-base on this specific task. If you're picking a base model for this example,
Mistral-7B is the best-validated choice.

## Run
An example for exllama
```bash
CUDA_VISIBLE_DEVICES=0,1 python examples/RAP/prontoqa/rap_inference.py --base_model exllama --model_dir your/path/to/llama --mem_map "[16, 22]" --depth_limit 6 --n_candidates 1 --temperature 0.0 # | tee debug_rap_chain.log
```

An example for llama2
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node 1 --master-port 1234  examples/RAP/prontoqa/rap_inference.py --base_model llama2 --model_dir your/path/to/llama --llama_size "7B"   --temperature 0.0 --n_candidates 1  --depth_limit 6
```

An example for llama3
Please set up `LLAMA3_CKPTS` or change this argument before running.
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node 1 --master-port 1234  examples/RAP/prontoqa/rap_inference.py  --base_model llama3 --model_dir $LLAMA3_CKPTS --llama_size "8B-Instruct"  --temperature 0.0 --n_candidates 1  --depth_limit 6
```
