from graph_enhanced_lm.config import read_env_file


def test_utf8_bom_utf16_quotes(tmp_path):
    body = "# c\nexport NEO4J_PASSWORD=\"abc123\"\nNEO4J_USER = neo4j\n"
    for enc, data in [("utf-8", body.encode()), ("bom", b"\xef\xbb\xbf" + body.encode()),
                      ("utf16", body.encode("utf-16"))]:
        p = tmp_path / f"{enc}.env"
        p.write_bytes(data)
        v = read_env_file(p)
        assert v["NEO4J_PASSWORD"] == "abc123" and v["NEO4J_USER"] == "neo4j", enc
