# Dataset Creation Guide (from zero to a graph)

This guide walks through every step from "I have no data" to "the data is in Neo4j".
The original dataset's schema **will differ** from the project's canonical schema; the
adapter + mapping YAML is where that difference is handled.

## 0. The canonical schema (target of every dataset)
| Table | Required columns | Meaning |
|---|---|---|
| `users.csv` | user_id, dataset_source, synthetic | one row per account |
| `posts.csv` | post_id, user_id, timestamp (ISO-8601 UTC), dataset_source, synthetic (+ text, language, source_labels) | one row per post |
| `interactions.csv` | interaction_id, interaction_type, timestamp + endpoints | RETWEETED (user→post), REPLIED_TO / QUOTED (post→post), MENTIONED (post→user) |
| `follows.csv` | source_user_id, target_user_id | only if the dataset has follow edges |
| `classification.csv` | post_id, text, source_labels, split | classifier-only datasets |
Full column definitions: [DATA_DICTIONARY.md](DATA_DICTIONARY.md).

## 1. Inspect a candidate dataset
Read its README and paper first and answer: *Does it have text? labels? user IDs? timestamps?
edges? What licence?* Then look at the files:
```powershell
python scripts/inspect_dataset.py --path data/raw/hatexplain/dataset.json --rows 2
```
This prints the format, columns, types, missing values and sample rows. Works for CSV, TSV,
JSON (object or array, large arrays are sampled), JSONL, XLSX and Parquet (needs `pyarrow`).

## 2. Does it match the objective?
Use the table in [DATASET_RESEARCH.md](DATASET_RESEARCH.md). If a field is missing, decide which
role the dataset can play (classifier vs graph) and write it in `capabilities:` of its mapping YAML.

## 3. Obtain the data
| Dataset | Command / procedure | Files expected in `data/raw/<name>/` |
|---|---|---|
| HateXplain | `python scripts/download_dataset.py --dataset hatexplain` (downloads from the official GitHub repo) | `dataset.json`, `post_id_divisions.json` |
| HASOC | Download from the year's page at hasocfire.github.io, accept its terms, then `python scripts/download_dataset.py --dataset hasoc --from-path <file>` | e.g. `en_2021_train.tsv` or `.xlsx` |
| TwiBot-22 | Email the authors (see their README) → receive Google Drive link → download → `python scripts/download_dataset.py --dataset twibot22 --from-path <folder>` | `user.json`, `tweet_0.json…`, `edge.csv`, `label.csv`, `split.csv` |
The script **copies** files (originals untouched) and writes `_download_manifest.json` with size
and SHA-256 so you can prove which file version you used. Never put credentials in code.
TwiBot-22 is tens of GB **[verify]**: put it on a drive with space and use the sampling settings.

## 4. Where files go
```
data/raw/<dataset>/        original files - NEVER edited
data/interim/              optional scratch space
data/processed/<dataset>/  canonical CSVs, predictions, validation report (generated)
data/synthetic/            generated synthetic data
```

## 5. Map fields to the canonical schema
Edit `configs/dataset_mappings/<dataset>.yaml`, never the Python code. Example (HASOC):
```yaml
columns:
  id: _id            # <- change to the real header in your file, e.g. text_id
  text: text
  coarse_label: task_1
  fine_label: task_2  # set to null if your file has no fine-grained column
```
TwiBot-22 relation names in `edge.csv` are mapped in `relation_map`. After receiving the data,
run `inspect_dataset.py` on `edge.csv` and compare the distinct `relation` values with the map.

## 6. Normalise IDs and timestamps (automatic in `preprocessing.py`)
* IDs → trimmed strings; `123.0` → `"123"`; empty → missing.
* Timestamps → ISO-8601 **UTC**. Naive timestamps are interpreted as UTC unless you change
  `assume_tz`. Twitter's `Wed Oct 10 20:19:24 +0000 2018` format is supported.
* Unparseable timestamps become missing and are **reported**, not silently dropped.

## 7. De-duplicate
Duplicates by `post_id`, `user_id`, `interaction_id` keep the first occurrence and are logged.
Interactions without an id get a deterministic SHA-1-based id, so re-runs produce the same id.

## 8. Missing values
| Field | Policy |
|---|---|
| post_id / user_id | row dropped (cannot be a node), counted in logs |
| timestamp | kept; excluded from temporal analysis with a warning |
| text | kept; classifier sees an empty string, consider excluding |
| labels | `UNKNOWN`, label_source `unlabeled` |

## 9-11. Map labels, preserve unknown labels
Mappings live in `configs/label_taxonomy.yaml → mappings`. Rules:
* every mapping is explicit (e.g. HASOC `PRFN → OFFENSIVE` is documented as an analytical choice);
* any source label without a mapping becomes `UNKNOWN` and a warning is logged;
* the original label is kept in `original_label` / `source_labels`.
HateXplain: majority vote of 3 annotators; a 3-way tie → `UNKNOWN`; a majority target community
on a hate/offensive post adds `COMMUNITY_TARGETED`.

## 12-13. Avoid leakage; split
* HateXplain's **official** train/val/test division is used when present.
* Otherwise `splitting.assign_splits` is used; if `group_id` (user or conversation) exists, all rows
  of one group land in one split. `preprocess_dataset.py` reports `groups_in_multiple_splits` (must be 0).
* Never tune thresholds on the test split.

## 14-15. Run the classifier and save predictions
```powershell
python scripts/classify_posts.py --dataset synthetic                        # source labels
python scripts/classify_posts.py --dataset twibot22 --backend transformer   # model labels
```
Output `data/processed/<name>/predictions.csv`: labels, primary_label, confidence, label_source,
model_name, model_version, predicted_at. The Transformer backend refuses to run if the model's
label names have no mapping. **Check the checkpoint's model card** (labels, licence, training data).

## 16. Prepare nodes and relationships
Handled by `graph/ingestion.py`: users → `(:User)`, posts → `(:Post)` + `POSTED`, hashtags parsed
from text → `USES_HASHTAG`, interactions → `RETWEETED/REPLIED_TO/QUOTED/MENTIONED`, predictions →
`HAS_LABEL` + `Post.primary_label`. Endpoints outside a sample become `placeholder: true` nodes.

## 17. Validate processed data
`preprocess_dataset.py` writes `validation_report.json` (fails on missing columns, duplicate keys,
unparseable timestamps, unknown interaction types; warns on dangling references).
`python scripts/validate_dataset.py --dataset synthetic` re-runs it.

## 18. Ingest into Neo4j
```powershell
python scripts/initialize_neo4j.py
python scripts/ingest_graph.py --dataset synthetic
python scripts/validate_graph.py
```

## 19. Add later batches
Put new raw files in place, re-run preprocessing/classification, then:
```powershell
python scripts/update_graph.py --dataset twibot22
```
Only new/changed posts (content hash) and new interactions are sent; MERGE makes repeats harmless;
human labels are never overwritten by model labels. Each run is an `(:IngestionBatch)` node.

## 20. Reproduce the whole pipeline
Pinned seed (`configs/project.yaml`), config files, download manifest (SHA-256), and the
`outputs/logs/*.json` run records together let anyone regenerate your numbers. Record the
`git` commit and the environment (`capture_environment()` output is stored with each evaluation).

## Example: creating a small dataset of your own
If your mentor gives you a CSV with columns `tweet_id, author, created, content, rt_of`:
1. Copy `configs/dataset_mappings/synthetic.yaml` to `mydata.yaml` and describe the files.
2. Write a 20-line adapter class in `dataset_adapters.py` (copy `SyntheticAdapter`), renaming
   columns to the canonical names and turning `rt_of` into RETWEETED interactions.
3. Register it in `GRAPH_ADAPTERS`, add `mydata` to the `--dataset` choices of the scripts you use, and run `preprocess_dataset.py --dataset mydata`.
Only use data you are permitted to use; never scrape protected content.
