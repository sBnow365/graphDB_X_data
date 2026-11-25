from graph_enhanced_lm.ui.config_edit import env_summary, set_yaml_scalar, update_env_value
from graph_enhanced_lm.ui.graph_viz import build_graph
from graph_enhanced_lm.ui.runner import display_command, script_command


def test_set_yaml_scalar_keeps_comments(tmp_path):
    (tmp_path / "temporal.yaml").write_text('# header\nelection_date: "2025-03-15"   # fictional\ntimezone: "Asia/Kolkata"\n')
    set_yaml_scalar("temporal", "election_date", "2016-11-08", config_dir=tmp_path)
    set_yaml_scalar("temporal", "timezone", "America/New_York", config_dir=tmp_path)
    text = (tmp_path / "temporal.yaml").read_text()
    assert 'election_date: "2016-11-08"   # fictional' in text and "# header" in text
    import yaml
    parsed = yaml.safe_load(text)
    assert parsed["timezone"] == "America/New_York" and parsed["election_date"] == "2016-11-08"


def test_set_yaml_scalar_plain_word_and_number(tmp_path):
    (tmp_path / "dataset.yaml").write_text("active_graph_dataset: synthetic      # comment\nx: 1\n")
    set_yaml_scalar("dataset", "active_graph_dataset", "russian_trolls", config_dir=tmp_path)
    set_yaml_scalar("dataset", "x", 0.6, config_dir=tmp_path)
    t = (tmp_path / "dataset.yaml").read_text()
    assert "active_graph_dataset: russian_trolls      # comment" in t and "x: 0.6" in t


def test_env_update_and_masking(tmp_path):
    env = tmp_path / ".env"
    env.write_text("NEO4J_PASSWORD=secret123\nLLM_BASE_URL=http://old/v1\n")
    update_env_value("LLM_BASE_URL", "https://new.trycloudflare.com/v1", env)
    update_env_value("LLM_MODEL", "qwen3:8b", env)
    text = env.read_text()
    assert "LLM_BASE_URL=https://new.trycloudflare.com/v1" in text and "LLM_MODEL=qwen3:8b" in text
    s = env_summary(env)
    assert "secret123" not in s["NEO4J_PASSWORD"] and "9 chars" in s["NEO4J_PASSWORD"]


def test_build_graph_colours_and_sizes():
    nodes = [{"id": "a", "pagerank": 2.0, "community": 3, "hate_label": "HATEFUL"},
             {"id": "b", "pagerank": 1.0, "community": 3},
             {"id": "p1", "kind": "Post", "label": "HATE"}]
    edges = [{"source": "a", "target": "b", "weight": 2}, {"source": "a", "target": "b", "weight": 1},
             {"source": "a", "target": "p1", "type": "POSTED"}, {"source": "x", "target": "a"}]
    g = build_graph(nodes, edges, color_by="hate_label")
    assert g["a"]["b"]["weight"] == 3.0 and "x" in g
    assert g.nodes["a"]["color"] == "#dc2626" and g.nodes["p1"]["color"] == "#dc2626"
    assert g.nodes["a"]["size"] > g.nodes["b"]["size"]
    g2 = build_graph(nodes, edges, color_by="community")
    assert g2.nodes["a"]["color"] == g2.nodes["b"]["color"]          # same community, same colour


def test_runner_commands():
    cmd = script_command("ask.py", "--no-llm", "a question")
    assert cmd[1].endswith("ask.py") and cmd[-1] == "a question"
    assert display_command("ask.py", "--no-llm", "a question") == 'python scripts/ask.py --no-llm "a question"'
