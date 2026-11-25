// Unique constraints on stable source identifiers.
// A uniqueness constraint also creates an index, so MERGE on these keys is fast.
// IF NOT EXISTS makes every statement safe to run repeatedly.
CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE;
CREATE CONSTRAINT post_id_unique IF NOT EXISTS FOR (p:Post) REQUIRE p.post_id IS UNIQUE;
CREATE CONSTRAINT hashtag_tag_unique IF NOT EXISTS FOR (h:Hashtag) REQUIRE h.tag IS UNIQUE;
CREATE CONSTRAINT label_name_unique IF NOT EXISTS FOR (l:ContentLabel) REQUIRE l.name IS UNIQUE;
CREATE CONSTRAINT community_key_unique IF NOT EXISTS FOR (c:Community) REQUIRE c.community_key IS UNIQUE;
CREATE CONSTRAINT batch_id_unique IF NOT EXISTS FOR (b:IngestionBatch) REQUIRE b.batch_id IS UNIQUE;
