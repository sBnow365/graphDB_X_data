# Data Dictionary (canonical tables, `data/processed/<dataset>/`)

## users.csv
| Column | Type | Description |
|---|---|---|
| user_id | string | stable source id (TwiBot-22 `u…`, synthetic `u_0000`) |
| username | string | public handle if provided (synthetic: `synthetic_user_…`) |
| dataset_source | string | synthetic / twibot22 |
| synthetic | bool | true for generated records |
| source_bot_label | string | TwiBot-22 `bot`/`human`. **Not a hate label** |
| synthetic_gt_community | string | planted community (synthetic only, for testing) |

## posts.csv
| Column | Type | Description |
|---|---|---|
| post_id | string | stable id |
| user_id | string | author |
| text | string | post text (synthetic: placeholder starting `[SYNTHETIC]`) |
| timestamp | ISO-8601 UTC string | creation time; empty if unknown |
| language | string | ISO code if given |
| source_labels | `;`-separated project labels | labels supplied by the dataset (empty for TwiBot-22) |
| conversation_id | string | root post of a thread (synthetic) |
| dataset_source, synthetic | | as above |

## interactions.csv
| Column | Type | Description |
|---|---|---|
| interaction_id | string | stable id (given or SHA-1 derived) |
| interaction_type | RETWEETED / REPLIED_TO / QUOTED / MENTIONED | |
| source_user_id | string | acting user (author of the retweet/reply/quote; author of the mentioning post) |
| source_post_id | string | reply/quote/mentioning post |
| target_post_id | string | retweeted/replied/quoted post |
| target_user_id | string | mentioned user |
| timestamp | ISO UTC | time of the interaction |

## follows.csv
source_user_id → target_user_id (follower → followed), dataset_source, synthetic.

## predictions.csv
| Column | Description |
|---|---|
| post_id | |
| labels | `;`-separated project labels |
| primary_label | display label by priority HATE > OFFENSIVE > SPAM > COMMUNITY_TARGETED > NORMAL > UNKNOWN |
| confidence | top-class probability (models only) |
| label_source | human / model / synthetic_ground_truth / unlabeled |
| model_name, model_version | classifier identity (HF commit hash when available) |
| predicted_at | UTC time |

## classification.csv (HateXplain / HASOC)
post_id, text, source_labels (mapped), original_label (raw), group_id (none for these datasets), split (train/val/test), language, dataset_source, targets (HateXplain).

## Label taxonomy
Defined in `configs/label_taxonomy.yaml`. Multi-label: HATE and COMMUNITY_TARGETED co-occur;
SPAM is a separate axis; UNKNOWN means unlabelled or unmapped.

## user_edges.csv (Hateful Users)
source_user_id → target_user_id (u retweeted v), edge_type RETWEETED_USER, weight (number of duplicate edges merged), dataset_source, synthetic.

## New columns
* `users.user_hate_label`: HATEFUL / NORMAL human user-level annotation (Hateful Users only; empty = not annotated).
* `posts.is_stub`: true for posts referenced by a retweet/reply but absent from the dataset (no text, no timestamp).
