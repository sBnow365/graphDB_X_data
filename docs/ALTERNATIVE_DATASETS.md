# Alternative datasets (no access application needed)

Use these if TwiBot-22 access is not granted. Both are free to download.

| | Hateful Users on Twitter | Russian troll tweets (NBC News) |
|---|---|---|
| Download | https://www.kaggle.com/datasets/manoelribeiro/hateful-users-on-twitter (free Kaggle login) | https://github.com/loretoparisi/russian_trolls_tweets_dataset (`tweets.csv`, `users.csv`) |
| Cite | Ribeiro et al., "Characterizing and Detecting Hateful Users on Twitter", ICWSM 2018 | NBC News, B. Popken, 14 Feb 2018; mirror by loretoparisi |
| Licence | shown on Kaggle (described as CC0 by users of the dataset; confirm on the page) | no explicit licence in the mirror: research use with citation; check before redistribution |
| Graph | User→User retweet edges (~2.2M) | users, tweets, retweets, replies, mentions |
| Labels | **human** user-level: hateful / normal (~5k users), others unannotated | none for content (classifier needed); all authors are IRA-linked accounts |
| Text / time | no raw text, no timestamps | text + timestamps (2016 US election period) |
| Project role | communities vs human hate labels (Parts 4–5, 7) | classification + temporal analysis around a real election (Parts 4–7) |

Never merge the two graphs: they have unrelated users. Load one at a time (empty graph in between).

## A. Hateful Users on Twitter
1. Download the ZIP from Kaggle, unzip it, then:
```powershell
python scripts/download_dataset.py --dataset hateful_users --from-path "C:\Users\RISHAV\Downloads\hateful-users-on-twitter"
python scripts/inspect_dataset.py --path data/raw/hateful_users/users_neighborhood_anon.csv --rows 2
Get-Content data\raw\hateful_users\users.edges -TotalCount 3
```
   Check that the CSV has columns `user_id` and `hate` (values hateful/normal/other). If names differ, edit
   `configs/dataset_mappings/hateful_users.yaml → columns`. The edge separator (space, comma or tab) is detected automatically.
2. Preprocess and analyse homophily offline:
```powershell
python scripts/preprocess_dataset.py --dataset hateful_users
python scripts/analyze_hateful_users.py --offline
```
   Output `outputs/metrics/hateful_users_analysis.json`: `p_retweet_hateful_given_hateful` vs `..._given_normal`
   and lift over a random baseline. (Do **not** run `classify_posts.py`: this dataset has no posts.)
3. Load into an **empty** graph, run GDS:
```powershell
# in configs/dataset.yaml set: active_graph_dataset: hateful_users
docker compose down -v ; docker compose up -d      # wipes the previous graph (regenerate synthetic any time)
python scripts/initialize_neo4j.py
python scripts/ingest_graph.py --dataset hateful_users   # ~2.2M edges: allow several minutes
python scripts/run_graph_algorithms.py
python scripts/run_community_analysis.py
python scripts/analyze_hateful_users.py                  # + concentration of hateful users in communities
```
   If GDS reports memory problems, raise `NEO4J_server_memory_heap_max__size` to `4G` in `docker-compose.yml`.
4. Questions (Part 7): `python scripts/ask.py "Which communities contain the most hateful users?"`.
   Cypher: `MATCH (u:User {user_hate_label:'HATEFUL'}) RETURN u.louvain_community, count(*) ORDER BY count(*) DESC`.
5. Not available: temporal analysis (no timestamps), post-level labels.

## B. Russian troll tweets (2016 US election)
1. Download:
```powershell
python scripts/download_dataset.py --dataset russian_trolls
```
   If it reports a Git-LFS pointer, download both files from the GitHub page manually and use `--from-path`.
2. Preprocess (default sample: 1 Sep–1 Dec 2016, max 30,000 tweets, in `configs/dataset_mappings/russian_trolls.yaml → sampling`):
```powershell
python scripts/inspect_dataset.py --path data/raw/russian_trolls/tweets.csv --rows 2
python scripts/preprocess_dataset.py --dataset russian_trolls
```
3. Classify with the Transformer (CPU is slow; Colab: guide Part 9.4):
```powershell
python scripts/classify_posts.py --dataset russian_trolls --backend transformer
```
   Stub posts (originals of retweets not in the data) have no text and are labelled UNKNOWN without the model.
4. Temporal analysis with the real election date (US presidential election, 8 Nov 2016):
```powershell
python scripts/run_temporal_analysis.py --source csv --dataset russian_trolls --election-date 2016-11-08 --timezone America/New_York
```
5. Graph:
```powershell
# configs/dataset.yaml: active_graph_dataset: russian_trolls ; configs/temporal.yaml: election_date: "2016-11-08", timezone: America/New_York
docker compose down -v ; docker compose up -d
python scripts/initialize_neo4j.py
python scripts/ingest_graph.py --dataset russian_trolls
python scripts/run_graph_algorithms.py
python scripts/run_community_analysis.py
python scripts/run_temporal_analysis.py --source neo4j
```
6. Interpretation rules: every author is IRA-linked by dataset definition, so communities here are structures
   *within a troll operation and the accounts it interacted with*, not a sample of normal users. Hate labels are
   model predictions (`label_source=model`). Retweeted-post authors are inferred from "RT @name:". Do not publish handles.

## Schema additions (both datasets)
| Element | Meaning |
|---|---|
| `(:User)-[:RETWEETED_USER {weight}]->(:User)` | user-level retweet graph (Hateful Users); included in the GDS projection only without date/label filters |
| `User.user_hate_label` | HATEFUL / NORMAL human annotation; `User.user_label_source = 'human_crowdsourced'` |
| `Post.placeholder = true` for stub posts | originals referenced by retweets/replies but not in the data; excluded from label shares |
| `configs/gds.yaml` | new weight `RETWEETED_USER`, new priority weight `hateful_user_share` |
