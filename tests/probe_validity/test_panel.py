import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments/paper1"))

import panel  # noqa: E402


def test_the_panel_has_the_earlier_drafts_eleven_models_and_llama_31():
    ids = {m.id for m in panel.PANEL}
    earlier = {"Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B",
               "meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3", "google/gemma-2-2b-it",
               "google/gemma-2-9b-it"}
    assert earlier <= ids and "meta-llama/Meta-Llama-3.1-8B-Instruct" in ids
    assert len(ids) == len(panel.PANEL)
    assert sum(m.role == "core" for m in panel.PANEL) == 12  # twelve full-sweep models; the 14B run is validity-only


def test_queue_ranks_are_unique_and_the_first_queue_models_have_none():
    ranks = [m.queue_rank for m in panel.PANEL if m.queue_rank]
    assert len(ranks) == len(set(ranks))
    first_queue = {m.id for m in panel.PANEL if not m.queue_rank}
    assert first_queue == {"Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "meta-llama/Meta-Llama-3.1-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"}


def test_a_gated_model_is_queued_only_when_its_weights_are_cached(monkeypatch, tmp_path):
    monkeypatch.setattr(panel, "CACHE", tmp_path)
    assert {m.id for m in panel.panel_queue()} == set()  # nothing is cached in an empty cache
    assert "Qwen/Qwen3-4B" in {m.id for m in panel.to_download()} and "meta-llama/Llama-3.2-3B-Instruct" not in {m.id for m in panel.to_download()}
    assert "meta-llama/Llama-3.2-3B-Instruct" in {m.id for m in panel.waiting_for_licence()}
    snap = tmp_path / "models--meta-llama--Llama-3.2-3B-Instruct/snapshots/abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    assert "meta-llama/Llama-3.2-3B-Instruct" not in {m.id for m in panel.panel_queue()}  # a config alone is not a copy of the weights
    (snap / "model.safetensors").write_text("x")
    assert "meta-llama/Llama-3.2-3B-Instruct" in {m.id for m in panel.panel_queue()}
    assert "meta-llama/Llama-3.2-3B-Instruct" not in {m.id for m in panel.waiting_for_licence()}


def test_sharded_weights_count_only_when_every_shard_is_present(monkeypatch, tmp_path):
    monkeypatch.setattr(panel, "CACHE", tmp_path)
    snap = tmp_path / "models--Qwen--Qwen3-8B/snapshots/abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "m-1.safetensors", "b": "m-2.safetensors"}}))
    (snap / "m-1.safetensors").write_text("x")
    model = panel.BY_ID["Qwen/Qwen3-8B"]
    assert not model.cached and model.downloadable and model not in panel.panel_queue()
    (snap / "m-2.safetensors").write_text("x")
    assert model.cached and not model.downloadable and model in panel.panel_queue()
    broken = tmp_path / "models--Qwen--Qwen3-4B/snapshots/abc"
    broken.mkdir(parents=True)
    (broken / "config.json").write_text("{}")
    (broken / "model.safetensors").symlink_to(tmp_path / "missing-blob")  # an interrupted download leaves a dangling link
    assert not panel.BY_ID["Qwen/Qwen3-4B"].cached


def test_the_validity_only_model_never_enters_the_core_queue_and_has_two_datasets():
    m = panel.BY_ID["Qwen/Qwen2.5-14B-Instruct"]
    assert m.role == "validity" and m.datasets == ("mmlu", "supergpqa_mfg")
    assert m not in panel.panel_queue()


def test_table_order_groups_by_family_then_size_and_names_are_stable():
    order = panel.table_order([m.id.replace("/", "__") for m in reversed(panel.PANEL)])
    shorts = [panel.short_name(i) for i in order]
    assert shorts[:3] == ["Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-7B"] and shorts[-1] == "Gemma-2-9B"
    assert panel.short_name("some/other-model") == "other-model"
    assert all(d in __import__("protocol").DATASET_TABLE for d in panel.DATASET_ORDER)
