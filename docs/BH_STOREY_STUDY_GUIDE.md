# Understanding the global seed-network BH and Storey analysis

This guide explains the statistical questions, calculations, graph definitions,
and results of the global seed-network comparison. It complements the
[operational documentation](QVALUE_CONNECTIVITY.md) and the
[results summary](../results/pipeline/kutsche_bh_gpu/fdr/seed_qvalue_comparison/README.md).
The numbers below describe the saved seed-network results, not a new analysis
of the full genome.

## 1. The main finding

We ranked directed interactions among **2,311 genes**, adjusted their p-values
using both BH and Storey, and examined how the network grows as the cutoff rises.

Two desired properties cannot be achieved together with a single cutoff:

- To put at least **95% of the genes in one connected component containing
  ZEB2**, the network has **508,408 directed connections** and ZEB2 has
  **1,178 distinct direct neighbors**.
- The largest selected network retaining **five direct ZEB2 neighbors** has
  **139 directed connections**, with **85 genes in its largest component**.
  That component contains ZEB2 and only about **3.7%** of the seed genes.

BH and Storey give the same networks at corresponding cutoffs in this analysis.
Changing adjustment methods changes the numerical threshold, but does not
resolve this tradeoff.

## 2. Where this step fits in the pipeline

```text
Expression measurements and sampling times
                  |
       Normalization and log1p transformation
                  |
   Directed interaction models for all seed pairs
                  |
       Saved raw p-values and test status
                  |
       Complete seed-family adjustment
             /                 \
     BH-adjusted p-values    Storey q-values
             \                 /
        Sort edges and apply successive cutoffs
                  |
     Count connections, components and ZEB2 neighbors
                  |
        Full tables, ZEB2 subsets and cutoff report
```

The computationally expensive model fits had already been performed. This
comparison reuses the saved p-values: it does not refit regressions, repeat
preprocessing, run permutations, or expand the genome search. The reporting
step uses the CPU; the earlier fitting stage can use the GPU.

The input is `all_pairs/overview/all_tests.csv` under
`results/pipeline/kutsche_bh_gpu/fdr/`. The report checks that the seed family
is complete and verifies the saved BH values against a fresh adjustment.

## 3. What each statistical test asks

For a predictor gene X and target gene Y, the model is:

```text
Y = a0 + a1*time + a2*time² + a3*X + a4*(time*X) + error
```

| Term | Role |
| --- | --- |
| `a0` | Baseline level |
| `a1*time + a2*time²` | A quadratic time trend in the target |
| `a3*X` | Association with the predictor at time zero |
| `a4*(time*X)` | How the predictor–target association changes with time |

The tested null hypothesis is **a4 = 0**, using a two-sided OLS t-test.
In this model the slope relating X to Y at time t is `a3 + a4*t`.
Thus, the test asks whether that slope changes with time, conditional on the
other model terms. It is not a test of every possible relationship between
the two genes.

The fitted data use 35 unaggregated WT samples: seven at each of days 0–4.
There are five fitted coefficients, giving 30 residual degrees of freedom
for an estimable full-rank fit. Expression was normalized using a median-ratio
procedure and transformed with log1p before fitting.

`X → Y` means X was the predictor and Y the target. Reversing them gives a
different regression. These arrows do not establish causal regulation, and
this particular model does not use a lagged predictor.

A small p-value is evidence against the model's null under its assumptions.
It is not the probability that the null is true, nor a measure of effect size.
The six-column export does not include the interaction coefficient, so it
cannot on its own tell us the sign or biological magnitude of an effect.

## 4. Why the testing family matters

The family is the complete set of directed, non-self seed pairs:

```text
G = 2,311
m = G × (G − 1) = 5,338,410 tests
```

The often-used shorthand `2,311 × 2,311` includes 2,311 self-comparisons.
Those are excluded here. Comparing a gene to itself does not provide an
identifiable version of this interaction test.

Adjustment is performed over the entire family before extracting ZEB2 rows.
The ZEB2 subset has `2 × 2,310 = 4,620` directed tests, but its reported values
remain **globally adjusted**. It is not corrected again as a 4,620-test family.

This distinction explains why earlier ZEB2-only results can differ even when
the underlying pairwise p-values agree: the denominator and ranks used for
multiple-testing correction differ. A q-value must always be interpreted
together with the family over which it was calculated.

The earlier adaptive workflow also used a discovered set of “core genes” to
start a genome-wide search. That core was a computational starting set, not a
designation of biological importance. Neither that adaptive family nor the
full genome is the family studied in this report.

## 5. Raw p-values, BH-adjusted values and Storey q-values

### Raw p-values

A raw p-value describes how unusual a test statistic at least as extreme as
the observed one would be under the null model. Testing millions of pairs
requires accounting for multiple comparisons rather than applying a raw
`p <= 0.05` rule independently to every pair.

### BH-adjusted p-values

BH orders the raw p-values from smallest to largest. At a chosen FDR level
alpha, it finds the largest rank i satisfying:

```text
p(i) <= alpha × i / m
```

It then selects all tests up to that rank. The adjusted value for each test
is the smallest BH level at which that test would be selected. In rank order:

```text
BH(i) = min(1, min over j >= i of [m × p(j) / j])
```

The cumulative minimum matters: an adjusted value is not always simply
`p(i) × m / i` for that row.

FDR means the expected fraction of false discoveries among the selected
tests, with the fraction defined as zero when nothing is selected. Under
valid p-values and suitable dependence assumptions, BH at a fixed level
controls this expectation. It does not promise that exactly that fraction
of this particular observed network is false.

### Storey q-values

Storey's method estimates **pi0**, the proportion of true null hypotheses
in the testing family. It uses the upper tail of the p-value distribution:

```text
pi0(lambda) = number of p-values >= lambda / [m × (1 − lambda)]
```

The intuition is that valid null p-values are approximately uniform, whereas
many non-null tests tend to have small p-values. This is an estimation model,
not proof that the fitted p-values have the required properties.

Here, lambda ranges from 0.05 to 0.95 in steps of 0.05. The implementation uses
the original R qvalue method's **closed-form bootstrap estimator** to choose
the estimate. This is an explicit choice, rather than the package's default
smoother. Lambda is an estimation parameter, not the final network cutoff.

“Bootstrap” here does not mean we resampled the biological samples or reran
the earlier permutation analysis. The calculation operates on saved p-values.
No additional q-value package was needed; the implementation was checked
against independent base-R calculations.

For this implementation (`pfdr=False`), the relationship is:

```text
Storey q = pi0 × BH-adjusted p
pi0 = 0.4718438636223145
```

Consequently, every estimable test retains its ordering. A BH cutoff of
0.00121117 corresponds to a Storey cutoff of approximately 0.000571485,
selecting exactly the same edges.

At the **same numerical cutoff**, Storey selects more edges here. At
**corresponding cutoffs**, the networks are identical. This is why the smaller
Storey threshold for 95% connectivity does not represent a sparser network.

Pi0 concerns null interaction hypotheses, not the fraction of genes without
biological function. Neither a Storey q-value nor a BH-adjusted p-value is a
posterior probability that an individual edge is false. No local FDR was
calculated in this report.

## 6. Why some tests failed

There are **6,930 non-estimable tests** and **5,331,480 estimable tests**.
The seed-family diagnostics identified three predictor genes—RPE65, KRT85
and IFNG—whose expression was nonzero at only one sampled day. For such a
predictor, `time*X` is a constant multiple of X. The regression cannot
separately estimate their coefficients: its design matrix is rank deficient.
Each affected predictor accounts for 2,310 outgoing tests.

These are missing statistical results, not nonsignificant measured p-values.
The analysis nevertheless keeps their planned tests in the correction family
by using internal **p=1 placeholders**. Their exported p-values, adjusted
values and cumulative counts remain blank; they do not become graph edges.

The same placeholders enter the primary Storey pi0 estimate, adding a mass
at p=1. Excluding them for a sensitivity calculation gives pi0 =
**0.46320471363798915**, compared with **0.4718438636223145** in the primary
analysis. The report discloses this difference rather than silently changing
the testing family.

## 7. How to read the network counts

| Quantity | Meaning |
| --- | --- |
| Directed connections | Selected ordered pairs; A→B and B→A count separately |
| Undirected pairs | Selected gene pairs ignoring direction; reciprocal links count once |
| Genes with connections | Genes with at least one selected incident edge |
| Largest component | Largest group joined by paths, ignoring arrow direction |
| ZEB2 component | The connected group containing ZEB2 |
| ZEB2 neighbors | Distinct genes directly linked to ZEB2 in either direction |

For example, in `ZEB2 → A → B → C`, ZEB2 has **one direct neighbor**, but
its weakly connected component has **four genes**. Component membership does
not imply a direct ZEB2 link. “Weakly connected” means we ignore arrow
direction when following paths; it is not strong directed connectivity.

The 95% target uses **all 2,311 seed genes** as its denominator:

```text
ceil(0.95 × 2,311) = 2,196 genes
```

We find the smallest observed cutoff at which a largest component contains
at least 2,196 genes and includes ZEB2. This means 95% node coverage within
one component. It does not mean 95% of possible edges exist or 95% of genes
are directly connected to ZEB2.

Edges enter when `adjusted value <= cutoff`. All edges tied at a value enter
together before counts are recorded. As the cutoff rises, edges and neighbors
can only be added, and components can merge but cannot split.

## 8. The actual cutoff results

| Selected network | BH cutoff | Storey cutoff | Directed edges | Largest component | ZEB2 neighbors |
| --- | ---: | ---: | ---: | ---: | ---: |
| First reaching 95% coverage with ZEB2 | 0.0012111737071 | 0.000571484881475 | 508,408 | 2,196 | 1,178 |
| First reaching five neighbors | 1.18947736691e-14 | 5.61247596495e-15 | 99 | 68 | 5 |
| Largest network retaining at most five neighbors | 2.86284269484e-14 | 1.35081475808e-14 | 139 | 85 | 5 |

Cutoffs are fractions: 0.001 is **0.1%**, not 0.001%. The 95%-coverage
cutoffs correspond to approximately 0.1211% for BH and 0.05715% for Storey.

At the 95% milestone:

- There are 451,661 undirected pairs among the 508,408 directed edges.
- 2,204 genes have edges; 2,196 are in the large component and eight are in
  other components. The remaining 107 genes are isolated.
- ZEB2 has 960 outgoing and 538 incoming links, giving 1,498 directed links.
- There are 320 reciprocal neighbors, so the distinct-neighbor count is
  `960 + 538 − 320 = 1,178`.

The five-neighbor set is **JARID2, NOVA2, PUM1, SOX4 and TCF4**. These five
links are outgoing from ZEB2. They are interaction-model candidates, not
experimentally confirmed regulatory targets.

The last row in the table refers to the last distinct observed adjusted
value retaining five neighbors. Raising a cutoff within a gap between
observed values would leave the graph unchanged.

Because the minimum cutoff reaching 95% already has 1,178 neighbors, every
higher cutoff also has at least that many. Therefore a single threshold
cannot simultaneously deliver 95% coverage and approximately five neighbors
in these results. This conclusion applies to both adjustments.

## 9. Reading the six-column tables

The cutoff comparison also includes fixed thresholds at each power of ten
from 10^-14 through 10^-5, alongside the original thresholds. These select
edges by adjusted value; the neighbor counts are reported outcomes. At
10^-14, BH selects 88 directed edges, a largest component of 66 genes and
four ZEB2 neighbors. Storey selects 118 directed edges, a largest component
of 76 genes and five ZEB2 neighbors. These are different graphs because the
same numerical cutoff is applied to two different adjustment scales.

| Column | Interpretation |
| --- | --- |
| `Genefrom` | Predictor gene |
| `Geneto` | Target gene |
| `p-value` | Raw two-sided interaction-test p-value |
| `q-value` | BH-adjusted p-value in the BH file; Storey q-value in the Storey file |
| `total_connections_at_or_below_q` | All selected directed seed-family edges at this row's threshold |
| `largest_component_nodes` | Largest weak component in that full thresholded seed graph |

The last two columns are **global cumulative graph statistics**, not the
degree of either gene in the row. All rows sharing an adjusted value have
the same counts. Failed rows occur last with blank statistics.

The ZEB2 subset preserves these columns unchanged. Its largest-component
count describes the full seed graph, not a graph constructed only from the
ZEB2 rows. For distinct neighbor identities, use
`zeb2_neighbors_at_milestones.csv` rather than counting directed rows.

## 10. What the results support—and what remains open

The results support a concrete descriptive conclusion: this model produces
many very small p-values, and broad seed-network coverage requires a large
ZEB2 neighborhood. The earlier concern about dense networks is therefore
still relevant. Storey's adjustment does not independently resolve it.

Small adjusted values alone do not establish that every selected edge is
biologically useful. Interpretation still depends on the model and its
residual assumptions, p-value calibration, sample structure, dependence
among tests, and effect sizes. Those questions were not revalidated by
sorting the saved p-values.

Selecting a threshold after examining connectivity is an exploratory graph
selection rule. The displayed adjusted values remain useful, but this step
does not establish a new error-control guarantee for that data-dependent
rule. Likewise, pooling tests from an adaptively chosen search is a separate
statistical question from correcting this complete, fixed seed family.

The practical next decision is which objective matters most:

- **Broad coverage:** retain the 95%-coverage graph, accepting many direct
  ZEB2 neighbors.
- **A small candidate list:** retain a strict neighborhood cutoff, accepting
  a small connected component.
- **Both views:** present the broad graph and a clearly labelled shortlist
  separately. A shortlist chosen by additional criteria is not the entire
  thresholded network.

Effect-size assessment, stability analysis or independent biological evidence
could help prioritize candidates, but none is supplied by the six-column
table alone. The 95% criterion is a stated design target, not evidence that
its cutoff is biologically optimal.

## 11. Files to study and share

All local paths below are relative to
`results/pipeline/kutsche_bh_gpu/fdr/seed_qvalue_comparison/`.

| File | Best use |
| --- | --- |
| `README.md` | Short methods and results overview |
| `cutoff_comparison.png` or `.svg` | See how connectivity and neighbors change with the cutoff |
| `cutoff_comparison.csv` | Exact counts at the requested thresholds and milestones |
| `zeb2_neighbors_at_milestones.csv` | Identify the neighbors and their directions |
| `bh/zeb2_connections.csv` | Inspect globally BH-adjusted ZEB2 links |
| `storey/zeb2_connections.csv` | Inspect the corresponding Storey values |
| `bh/connections_sorted.csv.gz` and `storey/connections_sorted.csv.gz` | Complete six-column tables |
| `summary.json` | Exact parameters, pi0 estimates, milestones and input provenance |
| `failed_tests.csv` | Identify non-estimable pairs |
| `export_verification.json` | Recorded export validation |

Shared copies may have method-prefixed filenames such as
`bh_zeb2_connections.csv` to distinguish attachments. The names above are
the local generated filenames. The plotted curves are sampled for display;
each method's `connectivity_events.npz` stores the exact events.

For a first reading, follow this order: this guide, the cutoff figure,
`cutoff_comparison.csv`, then the ZEB2 subset and milestone neighbor list.
The complete multi-million-row tables are for detailed downstream analysis.

## 12. Quick self-check

**Did we already have q-values?** We had BH-adjusted p-values, sometimes
loosely called q-values. This comparison adds Storey q-values with an
estimated pi0 and labels the methods explicitly.

**Does a q-value of 0.01 mean that edge has a 1% probability of being false?**
No. It is a multiple-testing selection measure, not an individual posterior
probability.

**Why are the Storey cutoffs smaller for the same network?** Every BH value
is multiplied by the same pi0, approximately 0.472.

**Does the 95% network give ZEB2 2,195 neighbors?** No. Its component has
2,196 genes, but only 1,178 are directly linked to ZEB2. Others are reached
through paths.

**Can we keep the 95% graph but report only five ZEB2 edges?** We can show
a separately labelled shortlist, but it would not describe all ZEB2 links
selected by that graph's cutoff.

**Are the failed tests evidence of no interaction?** No. Their coefficients
could not be identified. The p=1 values are internal bookkeeping only.

**Was the permutation procedure repeated?** No. This analysis adjusts the
saved regression p-values and measures graph connectivity.

## Method references

- [BH adjustment and dependence assumptions in R's p.adjust documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html)
- [Original Storey pi0 estimation implementation](https://github.com/jdstorey/qvalue/blob/master/R/pi0est.R)
- [Original q-value calculation](https://github.com/jdstorey/qvalue/blob/master/R/qvalue.R)
- [Local implementation and reproducibility commands](QVALUE_CONNECTIVITY.md)

The numerical findings in this guide come from the saved local report,
especially `cutoff_comparison.csv` and `summary.json`. No new fitting or
genome-wide analysis was performed to prepare this document.
