# Dataset Research

Research date: 30 Oct 2026. Items marked **[verified]** were checked against the official
repository / paper pages listed. Items marked **[verify]** are from documentation or memory
and must be confirmed when you obtain the files. No statistic here was invented; where a
number is unknown it is written as "unknown".

## 1. What the project needs from data

| Need | Why |
|---|---|
| Post text | content classification |
| Labels (hate/offensive/spam/target) | train/evaluate the classifier |
| User IDs | build User nodes |
| Timestamps | election-window analysis |
| Interaction edges (retweet/reply/quote/mention/follow) | communities, centrality |
| Election period coverage | the mentor's application |

**No single public dataset found satisfies all of these.** Hate-speech datasets have labels but
no network; network datasets have structure but no hate labels. Hence a **staged design**.

## 2. Candidate comparison

| Property | HateXplain | HASOC (2019–2024) | TwiBot-22 | Synthetic (this repo) |
|---|---|---|---|---|
| Official source | github.com/hate-alert/HateXplain **[verified]** | hasocfire.github.io/hasoc/ **[verified]** | github.com/LuoUndergradXJTU/TwiBot-22 ; twibot22.github.io **[verified]** | `scripts/generate_synthetic_data.py` |
| Paper | Mathew et al., AAAI 2021 **[verified]** | Mandl et al., FIRE overviews 2019–2021; later overviews by Modha, Satapara et al. **[verified titles]** | Feng et al., NeurIPS 2022 Datasets & Benchmarks **[verified]** | n/a |
| Access | public download | dataset pages per year; follow each year's terms **[verify per year]** | apply by email from institutional address stating institution, advisor, use case **[verified]** | none |
| License | MIT (repository) **[verified]** | check each year's page **[verify]** | research use; Twitter/X content-redistribution rules apply **[verified wording in README]** | project's own |
| Size | 20,148 posts **[verified, paper Table 2]** | varies by year/language (e.g. HASOC 2024 English: 1,776 items **[verified]**) | "largest Twitter bot detection benchmark" (exact counts: see their statistics page **[verify]**) | configurable (default 300 users) |
| Languages | English | English, Hindi, Marathi, German, Bengali, Gujarati, others by year **[verified]** | multilingual tweets **[verify]** | English placeholders |
| Labels | hatespeech / offensive / normal + target community + rationales **[verified]** | Task A: HOF / NOT; Task B: HATE / OFFN / PRFN **[verified]** | **bot / human** (NOT hate) **[verified]** | NORMAL/HATE/OFFENSIVE/SPAM/COMMUNITY_TARGETED |
| Text | yes (tokens) | yes | yes (tweet.json) | yes (placeholder) |
| User IDs | **no** | **no** (post id only) **[verify]** | yes | yes |
| Timestamps | **no** | **no** **[verify]** | yes (`created_at`) **[verify field name]** | yes |
| Edges | **no** | no (except conversational subtasks 2021–23 **[verify]**) | yes: edge.csv with relation types incl. follow, post, retweet, mention, reply, quote **[verify relation names]** | yes |
| Election-specific | no | no | not election-specific; collected before 2022 **[verified]** | fictional election date |
| Deleted/protected content | posts from Twitter + Gab; text already included | text included | text included in files | n/a |
| Classification suitability | **high** (English) | **high** (Hindi/code-mixed relevant to India) | low (no hate labels) | test only |
| Graph suitability | none | none | **high** | test only |
| Temporal suitability | none | none | **medium/high** | test only |
| Propagation-community suitability | none | none | **medium**: structure yes, hate labels must come from the classifier | test only |

### Other candidates (not implemented; investigate if needed)
* **Davidson et al. 2017 (ICWSM)** "hate speech / offensive / neither" tweets: classifier only.
* **OLID / OffensEval (SemEval-2019)** offensive + target type: classifier only.
* **TwiBot-20, Cresci-2017** bot datasets: graph structure, bot labels only.
* **Election tweet-ID collections** (various academic releases): usually tweet IDs only, which require
  re-hydration through the X API; current API pricing/limits make this impractical for a student
  project **[verify current X API terms]**. Do not scrape.

## 3. Recommendation (staged implementation)

| Stage | Dataset | Role | Status in code |
|---|---|---|---|
| 1 | **Synthetic** | develop + test the whole pipeline with known answers | implemented |
| 2 | **HateXplain** (primary classifier dataset) | evaluate a pretrained HateXplain checkpoint / optional fine-tuning | adapter implemented |
| 2b | **HASOC** (optional classifier dataset) | Hindi / code-mixed evaluation for an Indian election context | adapter implemented (column names configurable) |
| 3 | **TwiBot-22** (primary graph dataset) | real interaction graph, timestamps, Louvain, windows | adapter implemented; untested on real files |

**Why:** each dataset is used only for what it actually contains. The classifier trained/evaluated on
HateXplain labels TwiBot-22 tweets; those labels are stored as `label_source=model` and every
report must state they are model predictions on a different distribution (domain shift).

### Limitations of combining datasets
* Never join datasets by usernames or IDs: there is no documented link between them.
* TwiBot-22 tweets predate 2022 and are not tied to a specific election, so its time axis must be
  analysed with a **configurable** date. Choose a date relevant to the data's period and document
  your source, or present the window analysis as a method demonstration.
* Classifier errors propagate into community statistics; report classifier F1 next to every
  community finding.
* A first-N sample of TwiBot-22 (the adapter's default) is not random; report this.

## 4. Checklist to confirm with your mentor
- [ ] Title: keep "Scientific Literature" or adopt the subtitle / a new title?
- [ ] Is a method demonstrated on synthetic + TwiBot-22 + HateXplain acceptable, given that no public dataset combines hate labels, network edges and an Indian election period?
- [ ] Which election (country, date, source) should the window analysis reference?
- [ ] Is English-only classification acceptable, or is HASOC Hindi evaluation required?
- [ ] May you apply for TwiBot-22 access using the mentor as advisor?
- [ ] Is SPAM required on real data (no spam-labelled tweet dataset is integrated yet)?
- [ ] Are model-generated labels acceptable for community analysis if their F1 is reported?
