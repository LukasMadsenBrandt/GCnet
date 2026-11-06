# Seed-to-genome families and multi-anchor reports

There are three distinct fixed testing families:

| Family | Definition | Current planned tests |
| --- | --- | ---: |
| Seed network | Both directions between 2,311 retained seed genes, no self-pairs | 5,338,410 |
| Seed-to-genome | Every ordered pair with at least one endpoint among those seeds | 206,816,012 |
| Entire genome | Every ordered non-self pair among 45,902 retained genes | 2,106,947,702 |

The retained counts determine the denominators, rather than the approximate
2,310 and 45,000 labels. No connected-component selection reduces the seed list.

## Priority reports from the existing seed network

Generate ZEB2, ZBTB18 and MECP2 reports without fitting any new regressions:

```bash
python3 scripts/experiments/seed_anchor_report.py \
  --run-dir results/pipeline/kutsche_bh_gpu/fdr \
  --output-dir results/reports/seed_anchor_qvalues \
  --anchors ZEB2 ZBTB18 MECP2 --cutoff 1e-6
```

The script validates the complete saved family and its BH values, then
calculates Storey using the same convention as the earlier comparison. Both
methods use all seed tests; the anchor subsets are not adjusted separately.
Numba is required for the streaming graph calculations (included in
`requirements-genome.txt`). Each selected directed-edge file is sorted by
adjusted value; each distinct-neighbor file is sorted by its smallest selected
directional adjusted value. Incoming and outgoing directions are explicit.

The generated `anchor_cutoff_comparison.csv` gives all three anchors' component
sizes and incoming, outgoing, reciprocal and distinct neighbors at the fixed
thresholds from 10^-14 through 0.05, including the earlier intermediate levels.
Component size includes the anchor; direct-neighbor counts exclude it.

The saved seed results at adjusted cutoff 10^-6 are:

| Anchor | Method | Outgoing | Incoming | Distinct neighbors | Anchor component |
| --- | --- | ---: | ---: | ---: | ---: |
| ZEB2 | BH | 432 | 34 | 450 | 1,557 |
| ZEB2 | Storey | 480 | 51 | 508 | 1,645 |
| ZBTB18 | BH | 466 | 79 | 471 | 1,557 |
| ZBTB18 | Storey | 515 | 99 | 524 | 1,645 |
| MECP2 | BH | 0 | 39 | 39 | 1,557 |
| MECP2 | Storey | 0 | 44 | 44 | 1,645 |

Reciprocal neighbors explain why the distinct count is smaller than the sum
of outgoing and incoming counts. All component sizes refer to the full weakly
connected seed graph, not just to anchor edges. The two adjustment scales
select different graphs at the same numeric cutoff.

At this cutoff MECP2's selected links are all incoming, unlike the predominantly
outgoing neighborhoods of ZEB2 and ZBTB18. These directions identify regression
predictor/target roles; they do not establish causal regulation. Absence of
selected outgoing links is specific to this model, family and cutoff.

## Expanded fixed seed family

For S seed genes and G retained genome genes, both directions give:

```text
S × (G − 1) + (G − S) × S = S × (2G − S − 1)
```

The first block contains seed→genome tests, including every seed→seed test.
The second contains only nonseed→seed tests. Thus seed→seed pairs are included
once, self-tests are excluded, and nonseed→nonseed tests are not part of this
family. Opposite directions remain different tests.

Both BH and Storey must be recalculated across this entire union. Do not
concatenate already-adjusted seed and genome values, and do not count the old
seed family again. The new run fits the union consistently, including its seed
pairs; it does not import or append old adjusted tables. Its raw files keep
the same global pair-ID encoding as the full-genome workflow.

Use [the production seed-to-genome configuration](../configs/production/seed_genome.kutsche.fdr_gpu.yml):

```bash
python3 scripts/pipeline/run_genome_fdr.py \
  --config configs/production/seed_genome.kutsche.fdr_gpu.yml --plan

# On the GPU machine: a bounded pilot, then the complete fitting stage.
python3 scripts/pipeline/run_genome_fdr.py \
  --config configs/production/seed_genome.kutsche.fdr_gpu.yml --pilot-shards 4
python3 scripts/pipeline/run_genome_fdr.py \
  --config configs/production/seed_genome.kutsche.fdr_gpu.yml --stage fit

# Once fitting is complete, postprocess on CPU with the same data/checkpoints.
python3 scripts/pipeline/run_genome_fdr.py \
  --config configs/production/seed_genome.kutsche.fdr_gpu.yml \
  --backend cpu_numpy --stage report
```

The new run name keeps this family separate from the full-genome run. The
same GPU fitting, progress, checkpoints and global disk-backed adjustment
apply; see [the operational guide](GENOME_FDR_PIPELINE.md). The numeric payload
estimate for raw, sorted and adjusted data is about 19.9 GB before compression
and Parquet overhead, plus external-sort scratch. This is not a free-space
guarantee; use the pilot and provision additional headroom. No complete
seed-to-genome or full-genome fitting was launched to prepare these changes.

## Reporting multiple anchors and outside-seed genes

Both production configurations specify:

```yaml
gene_of_interest: ZEB2
analysis:
  report_anchors: [ZEB2, ZBTB18, MECP2]
```

`anchor_cutoff_comparison.csv` includes one row per method, anchor and cutoff.
Global edge/component counts repeat across anchors; anchor-specific component
and neighbor counts can differ. Outside-seed columns count active genes,
anchor-component genes and distinct anchor neighbors outside the **retained**
reference list. Isolated anchors have component size one, even without edges.
`anchor_connections_all.csv` contains all estimable directed anchor pairs,
with an anchor column and both global adjustments. Pairs linking two report
anchors appear once in each anchor view, but only once in the testing family.

The original primary-anchor files remain for compatibility. Their legacy
`zeb2_*` columns refer to `gene_of_interest`. Report anchors can be changed on
resume without refitting completed tests; the reporting stage scans the
existing adjusted files again. Full-genome family membership is unaffected
by the reference seed file used for reporting.

## Interpreting edge count multiplied by cutoff

For false discoveries V and total discoveries R, FDR concerns
`E[V / max(R,1)]` under the method's assumptions. It is not a guarantee that
`E[V | observed R] = cutoff × observed R`. The tables retain that product as
`directed_edges_times_cutoff`, labelled as a descriptive calculation.

For example, at 10^-6 the seed-family products are 0.063745 for BH and
0.079805 for Storey. These do not prove zero false edges. Even an established
expected false-edge count below one would still allow individual networks
to contain errors. Threshold selection from these results remains exploratory,
and model calibration and biological validation remain separate questions.
See the [R statistical documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html)
for the distinction between false discovery rate and family-wise error rate.

The expanded seed family includes the seed tests requested for pooling; that
pooling is part of its correction, not a fourth independent analysis. All three
anchors are also reported for the complete genome family.
