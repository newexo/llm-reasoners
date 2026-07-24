# Blocksworld

## Preparation

1. Download the test cases.
   - **If you cloned `llm-reasoners` from github**: If you have cloned with the `--recursive` option, there should be a `LLMs-Planning` folder in the root directory already. Otherwise, you could run `git submodule update --init` to get it.
   - **If you installed `llm-reasoners` with pip**: Please clone the `LLM-Planning` repo and add it to the environment variable:
   ```bash
   git clone https://github.com/karthikv792/LLMs-Planning.git
   cd LLMs-Planning && git checkout fc638a1aff7df3fe7a1a1d289fa2c04cc24dc284
   echo "export PLANBENCH_PATH=$(pwd)" >> ~/.bashrc && source ~/.bashrc
   ```

2. Set up `VAL` for evaluation. Build it from the official [KCL-Planning/VAL](https://github.com/KCL-Planning/VAL) repository (the plan validator originally developed at the University of Strathclyde / King's College London, BSD-3-Clause licensed) — not the copy vendored inside `LLMs-Planning`, which carries stale/inconsistent license files and hasn't been updated since 2023:
   ```bash
   git clone https://github.com/KCL-Planning/VAL.git && cd VAL
   make validate
   ```
   Only `validate` is needed here (`parser`/`tan` are unused by this repo).

3. Assign the path of the directory containing the built `validate` executable to the environment variable VAL: `export VAL=/path/to/VAL`

## Data Description

The full [Blocksworld](https://arxiv.org/abs/2305.15771) datasets contain 602 samples.

There are two settings to run Blocksworld: Hard (v1) and Easy (v2).

- In Easy setting, we assume the minimum number of actions for each case is known. Leveraging this information, we use demonstration cases that share the same minimum number of actions as the test case. E.g., to solve a problem that can be solved with at least 6 steps, we will use other 6-step problems as the in-context demonstration. For each group of cases, we randomly select 10 cases to create a pool of demonstration cases, leaving the remaining cases as the test set (540 cases in total). During inference, we randomly sample 4-shot demonstration cases from this pool and utilize them to formulate prompts. 

- In the Hard setting, we randomly select 10 cases from the full dataset to form a demonstration pool and subsequently exclude these cases from the test set (590 cases in total). During inference, we randomly sample 4-shot demonstration cases from this global pool, irrespective of the minimum number of actions required for the test case.

In the paper [Reasoning-via-Planning](https://arxiv.org/pdf/2305.14992), we report results of both settings in Table 4. In the paper [LLM Reasoners](https://arxiv.org/abs/2404.05221), we report the results on the hard setting (v1) in Table 3 and Figure 6.