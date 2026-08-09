# Notes & limitations

### Dialect-code spelling in `precompute_probabilities.py`

That module spells Alexandria / Riyadh / Sfax as `ALX` / `RIY` / `SFA`, while the
rest of the codebase and `data/MADAR/MADAR.tsv` use `ALEX` / `ARI` / `SFX`. The
difference has no effect on results: only the six core dialects (`BEI`, `CAI`,
`DOH`, `RAB`, `TUN`, `MSA`) have CODA/CAPHI resources, so the other 20 codes only
ever key empty `dialect_dict` entries and carry zero probability mass regardless of
spelling.

### `P_CAPHI_given_ORTHO_by_dialect.{json,tsv}` are not bit-reproducible

`precompute_probabilities.py` normalizes `P(CAPHI|ORTHO)` with
`sum([... for caphi in list(set(...))])`. Python's set iteration order varies with
`PYTHONHASHSEED`, so the summation order — and hence the last floating-point ULP —
changes between runs. Values are stable to ~1e-10; set `PYTHONHASHSEED=0` for an
exactly reproducible file. Every other probability artifact is byte-stable.

### `AGS_extraction` runtime

`AGS_extraction.AGS_extraction` computes an affix-aware DP distance for every
aligned word pair across up to 26 dialects. The per-character substitution cost is
cached across the whole run, so it amortizes, but a full-corpus run still takes
**hours**. Run it on a slice first to sanity-check the wiring.

### License

`pyproject.toml` does not declare a code license. Bundled corpora under `data/`
keep their own upstream licenses (`LICENSE.txt` next to each). Add a top-level
`LICENSE` before distributing.
