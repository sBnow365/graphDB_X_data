# Ethics and Limitations

## Safeguards implemented
| Risk | Safeguard |
|---|---|
| Misidentifying people as malicious | outputs speak of *candidate communities exhibiting hate/spam propagation patterns*; prompts forbid claims of coordination/intent; ranking labelled "for human review" |
| Model-label errors | `label_source`, model name/version stored per label; classifier F1 must be reported next to findings |
| Overwriting human annotations | ingestion never replaces a `human` primary label with a model label |
| Synthetic presented as real | `synthetic=true` on every record, `[SYNTHETIC]` text prefix, notes in every RAG answer and report |
| Privacy | data minimisation (no private data, only fields needed), post text truncated in evidence, not written to logs (`privacy.log_post_text: false`) |
| Licence violations | download only from official sources; manual procedures for access-controlled data; raw data git-ignored and never redistributed |
| Deleted/protected content | never scraped; X API optional and only within its terms |
| Destructive or exfiltrating queries | read-only Text-to-Cypher with layered validation |
| Political profiling | no political-alignment inference implemented; if ever explored, separate module, documented method and uncertainty, never stored as fact about a person |
| Causal over-claiming | temporal outputs carry a causality caveat |

## Known limitations
* No public dataset combines hate labels, interaction edges and an election period; findings on
  TwiBot-22 rely on a classifier trained on a different dataset (domain shift).
* HateXplain is English (Twitter + Gab); Hindi/code-mixed content needs HASOC-trained models.
* Hate-speech classifiers show bias against some dialects and identity terms (documented in the
  literature, e.g. HateXplain's bias analysis). Measure per-group errors where target labels allow.
* Louvain results vary with projection weights and between runs; report sensitivity.
* TwiBot-22 first-N sampling is not random.
* Rule-based intent parsing covers limited phrasings; unknown phrasings depend on LLM quality.
* Synthetic data has planted structure; good synthetic results do not validate real-world performance.

## Human review
Any statement about a specific real account or community must be checked by a person against the
underlying posts, with uncertainty stated. Do not publish account identifiers from real data in the report.
