# GSM8K (Qwen2.5-0.5B-Instruct, n=20, greedy, max 256 new tokens) – band-filter intervention

Bands on the residual stream along tokens: slow (EMA .9, always kept) / low (period ~9-60 tok) / high (<9 tok).
Mid-layer energy share of |h|^2 on generated tokens: slow 0.50, low 0.17, high 0.14.
Coherence = NLL of the generated text under the UNMODIFIED model (low NLL + high rep4 = repetition loop, not coherence).

| condition | acc | NLL | rep4 | finished | non-ASCII |
|---|---|---|---|---|---|
| base | 0.25 (±0.10) | 0.163 | 0.20 | 0.30 | 0.0002 |
| mid low x0.5 | 0.25 | 0.231 | 0.16 | 0.40 | 0 |
| mid high x0.5 | 0.20 | 0.233 | 0.18 | 0.30 | 0 |
| mid low x0 | 0.00 | 1.260 | 0.22 | 0.55 | 0 |
| mid high x0 | 0.00 | 1.632 | 0.25 | 0.20 | 0 |
| all-layers low x0.5 | 0.00 | 0.208 | 0.99 | 0.00 | 0 |
| all-layers high x0.5 | 0.00 | 1.310 | 0.93 | 0.05 | 0 |

Raw numbers: scores.json; baseline CoT spectrum: spectrum_report.md / spectrum_summary.json.
Caveat: n=20 (accuracy SE ~0.10); 70% of baseline answers are truncated at 256 tokens.
