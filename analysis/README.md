# Result analysis

Open the notebooks in numeric order and select the `vinh_vllm` Python kernel:

1. `01_data_quality_and_response_validity.ipynb` — completeness, parse failures,
   response length, and model differences in format reliability.
2. `02_position_and_permutation_bias.ipynb` — A–D position preference, χ² tests,
   and consistency across the four candidate permutations.
3. `03_demographic_selection_fairness.ipynb` — exposure-adjusted gender, race,
   and age selection rates with multiplicity-corrected screening tests.
4. `04_paired_ablations_and_fairness_instruction.ipynb` — matched decision flips,
   group-cluster bootstrap intervals, and exact paired intervention tests.

Install or refresh the analysis dependencies with:

```bash
source /apps/local/conda_init.sh
conda activate vinh_vllm
pip install -e '.[analysis]'
```

`common.py` streams the nested JSONL files into compact decision and candidate
tables. It stores a fingerprinted Parquet cache in `analysis/.cache/` and rebuilds
it automatically whenever an output file changes. The cache and optional exported
figures are ignored by Git.

The tests are exploratory diagnostics. The notebooks explicitly distinguish
format/position effects, exposure-adjusted associations, and paired ablations;
none should be interpreted alone as proof of causal discrimination.
