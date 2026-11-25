# Temporal and Election-Window Analysis

* Stored timestamps: UTC. Window boundaries: local midnight in `configs/temporal.yaml → timezone`
  (default Asia/Kolkata), converted to UTC. Intervals are half-open `[start, end)`.
* `election_date` must come from you (a verified official source) or stay the synthetic fictional
  default. It is never assumed. Override per run: `--election-date YYYY-MM-DD`.
* Default windows: `pre_30_to_11`, `pre_10_to_1`, `election_day`, `post_1_to_10` (editable).
* Posts without timestamps are counted, logged and excluded.

Outputs (`run_temporal_analysis.py`): `window_comparison.csv/.md` (posts, posts/day, label shares per
window), `daily_label_counts.csv` + figure, `community_window_activity.csv` (Neo4j mode).
Temporal projections: `create_gds_projection.py --start … --end …` then run Louvain on each window's
projection to study changing membership; compare centrality across windows by writing to different
properties (e.g. `pagerank_pre10`).

**Interpretation rule:** "observed hate share rose from X to Y between windows" is a description.
"The election caused…" requires a causal design (comparison periods, other events, platform changes)
that this project does not have. The code prints this caveat with every temporal result.
