# Optional X API Integration (not implemented)

The pipeline is complete without X API access. If you consider it later, verify the **current** X
developer terms, pricing tiers, rate limits, historical-search availability and content-redistribution
rules on the official developer site before writing code **[verify; these change frequently]**.
Rules for this project: credentials only in `.env`; never access protected/private content; never
bypass limits; respect deletions (remove deleted posts on re-sync); store only fields you need;
implement it as a new adapter producing the canonical tables so the rest of the system is unchanged.
