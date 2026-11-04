# Global seed-network BH and Storey comparison

This experiment produces a six-column global connection table and the
ZEB2 subset, comparing conventional BH-adjusted p-values with Storey q-values.
It uses the **complete saved seed family**, not the local ZEB2 family or the
adaptive core-to-genome family. No regression, preprocessing or GPU run is needed.

```bash
python3 scripts/experiments/qvalue_connectivity.py \
  --run-dir results/pipeline/kutsche_bh_gpu/fdr
```

Default output: `results/pipeline/kutsche_bh_gpu/fdr/seed_qvalue_comparison/`.
Use `--output-dir` for a different report directory. The input must include
`manifest.json`, `run.lock`, `all_pairs/overview/summary.json`, and the complete
`all_pairs/overview/all_tests.csv`. The complete seed stage can be reported even
if the subsequent genome expansion is unfinished. An active pipeline writer
prevents reporting. Saved pair identities, row counts, failure flags and BH
values are verified before the report is generated.

The retained set contains 2,311 genes and 5,338,410 directed non-self tests
(`G*(G-1)`). Self-comparisons are excluded; they are not identifiable interaction
tests. Both methods use exactly this test family.

## Statistical methods

- **BH:** recompute the adjustment from all raw p-values to validate the saved
  values. Failed tests remain in the denominator as p=1 placeholders and have
  no reported adjusted value, matching the pipeline's existing convention.
- **Storey:** estimate pi0 using the original R `qvalue` package's **closed-form
  bootstrap** equations, with lambda=0.05,0.10,...,0.95. Use `pfdr=False` and do
  not estimate local FDR. This is an explicit choice of the bootstrap estimator,
  not the smoother default of `py-qvalue` or R `qvalue`. No extra q-value package
  is required. Nonpositive estimates and truncated p-value inputs raise errors.
- **Failed tests:** the primary Storey calculation includes the same p=1
  placeholders in both the denominator and pi0 estimation. They add an atom at
  1 rather than observed evidence. The report also gives a sensitivity estimate
  of pi0 calculated from estimable tests only; it does not silently change the
  primary testing family. The placeholder policy and pi0 estimates must be
  disclosed when interpreting the result.

For this standard Storey calculation, q-values equal BH-adjusted values times
the estimated pi0. The ranking and networks at corresponding cutoffs are the
same; numerical cutoffs shift. Storey's estimate is not an independently
validated biological fraction of genes or interactions.

References: [original pi0 estimator](https://github.com/jdstorey/qvalue/blob/master/R/pi0est.R),
[original q-value calculation](https://github.com/jdstorey/qvalue/blob/master/R/qvalue.R).

## Tables and connectivity

Each method has `connections_sorted.csv.gz` with exactly six columns:

1. `Genefrom`
2. `Geneto`
3. `p-value`
4. `q-value` (BH-adjusted p-value in the BH table)
5. `total_connections_at_or_below_q`
6. `largest_component_nodes`

The table is sorted by increasing adjusted value. Complete tied-value groups
are added before evaluating graph counts, using **q <= cutoff**. All rows in a
tie get the same counts. Failed tests appear last with blank statistics and
counts, and their identifiers are also listed in `failed_tests.csv`.

Each method's `zeb2_connections.csv` is the sub-list involving ZEB2 in either
direction, using the **same global adjustments and network counts**. It is not
adjusted again as a smaller family.

Components are weakly connected: links are traversed in either direction.
Reciprocal links count as two directed connections but one undirected pair and
one ZEB2 neighbor. The 95% target means one component containing at least
ceil(0.95*2311)=2,196 seed genes **and ZEB2**, not 95% edge density.

`cutoff_comparison.csv` reports the requested fixed thresholds and these exact
milestones for each method:

Fixed thresholds are 10^-14 through 10^-5 (each power of ten), followed by
0.0001, 0.001, 0.002, 0.005, 0.01 and 0.05. These thresholds apply to
BH-adjusted p-values or Storey q-values, respectively, using `value <= cutoff`;
they are not raw p-value or neighbor-count thresholds.

- The smallest cutoff achieving the 95% target and including ZEB2.
- The first cutoff with at least five distinct ZEB2 neighbors.
- The last distinct observed cutoff retaining at most five ZEB2 neighbors.

The first and last milestones need not correspond to compatible networks. Ties
can also skip directly over five neighbors. Missing milestones are recorded as
null in `summary.json`. The cutoff selection is exploratory and does not itself
establish an additional FDR guarantee or validate the original p-values.

`zeb2_neighbors_at_milestones.csv` gives the neighbor identities and directions.
The plots show sampled curves; `connectivity_events.npz` retains exact counts
for every observed adjusted-value group. Progress is printed during reading,
incremental union-find component tracking, and table export. NumPy/SciPy and
pandas handle the statistics and files; installed Numba optionally accelerates
the graph traversal. Reporting uses CPU and writes only the new report files.

Tests compare pi0 and adjusted values to an independent base-R calculation and
BH `p.adjust` reference, and component counts to SciPy, including ties,
reciprocal directions, failed tests, disconnected graphs and incompatible goals.

```bash
python3 -m pytest tests/test_qvalue_connectivity.py -q
```
