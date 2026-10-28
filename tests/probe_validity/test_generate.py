import math

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from probe_validity.generate import Runner, content_step, label_token_ids, read_jsonl, run_condition  # noqa: E402

PROMPTS = [
    "Say A.",
    "Question: which of these is a prime number? Answer with a letter.\nA) 4\nB) 6\nC) 7\nD) 9",
    "Hello there, how do you do on this fine day when the weather is rather pleasant?",
    "Yes or no?",
    "Name a colour.\n\nThen explain why in a sentence that is longer than the others in this list of prompts.",
    "Q",
]


@pytest.fixture(scope="module")
def tok():
    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", padding_side="left", local_files_only=True)
    except Exception:  # tokenizer not cached on this machine
        pytest.skip("Qwen2.5 tokenizer is not cached")


def tiny_model(tok, attn: str):
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                      tie_word_embeddings=False, attn_implementation=attn)
    model = Qwen2ForCausalLM(cfg).float()
    # The Hub defaults of Qwen2.5-Instruct: sampling and a repetition penalty that a naive greedy call would inherit.
    model.generation_config.update(do_sample=True, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05,
                                   eos_token_id=[151645, 151643], pad_token_id=151643)
    return model


@pytest.fixture(params=["sdpa", "eager"])
def runner(request, tok):
    return Runner("tiny", device="cpu", dtype="float32", attn=request.param, model=tiny_model(tok, request.param), tok=tok)


def test_hub_sampling_defaults_are_reset(runner):
    gc = runner.effective_config(8)
    assert gc["do_sample"] is False and gc["repetition_penalty"] == 1.0 and gc["num_beams"] == 1
    assert gc["temperature"] is None and gc["top_p"] is None and gc["top_k"] is None and gc["max_new_tokens"] == 8
    sampled = runner.effective_config(8, {"temperature": 0.7, "top_p": 0.8})
    assert sampled["do_sample"] is True and sampled["temperature"] == 0.7 and sampled["top_k"] is None
    meta = runner.meta(8)
    assert meta["truncation"] is False and meta["add_special_tokens"] is False and meta["padding_side"] == "left"
    assert meta["generation_config"]["repetition_penalty"] == 1.0


def test_greedy_is_argmax_of_raw_logits_and_label_logprobs_are_raw(runner):
    labels = ("A", "B", "C", "D")
    rows = runner.generate_batch(PROMPTS, 3, labels=labels)
    for prompt, row in zip(PROMPTS, rows):
        enc = runner.tok(runner.render(prompt), return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            logp = torch.log_softmax(runner.model(**enc).logits[0, -1].float(), -1)
        assert row["ids"][0] == int(logp.argmax())
        assert abs(row["lp"][0] - float(logp[row["ids"][0]])) < 1e-3
        for lab in labels:
            ids = label_token_ids(runner.tok, lab)
            assert ids and abs(row["label_lp"][lab] - float(torch.logsumexp(logp[ids], 0))) < 1e-3


def test_batched_equals_single_prompt_generation(runner):
    batched = runner.generate_batch(PROMPTS, 6)
    for prompt, row in zip(PROMPTS, batched):
        single = runner.generate_batch([prompt], 6)[0]
        assert single["ids"] == row["ids"] and single["finish"] == row["finish"]
        assert max(abs(a - b) for a, b in zip(single["lp"], row["lp"])) < 1e-3


def test_nan_logits_raise_instead_of_becoming_a_reply(tok):
    model = tiny_model(tok, "eager")
    with torch.no_grad():
        model.lm_head.weight.fill_(float("nan"))
    runner = Runner("tiny", device="cpu", dtype="float32", attn="eager", model=model, tok=tok)
    with pytest.raises(RuntimeError, match="NaN logits"):
        runner.generate_batch(PROMPTS[:2], 3)


def test_plan_is_fixed_complete_and_longest_first(runner):
    batches = runner.plan(PROMPTS, 4)
    assert batches == runner.plan(PROMPTS, 4)
    assert sorted(i for b in batches for i in b) == list(range(len(PROMPTS)))
    lengths = [len(runner.tok.encode(runner.render(p), add_special_tokens=False)) for p in PROMPTS]
    flat = [i for b in batches for i in b]
    assert [lengths[i] for i in flat] == sorted(lengths, reverse=True)


def test_run_condition_resumes_and_redoes_only_the_cut_batch(runner, tmp_path):
    out = tmp_path / "c.jsonl"
    qids = [f"q{i}" for i in range(len(PROMPTS))]
    first = run_condition(runner, "c", qids, PROMPTS, out, max_new_tokens=4, batch_size=2, labels=("A", "B"), log=lambda *_: None)
    assert [r["qid"] for r in first] == qids
    lines = out.read_text().splitlines()
    out.write_text("\n".join(lines[:-1]) + "\n" + lines[-1][:20])  # an interrupted write: last row cut short
    calls = []
    original = runner.generate_batch
    runner.generate_batch = lambda prompts, *a, **k: (calls.append(len(prompts)), original(prompts, *a, **k))[1]
    again = run_condition(runner, "c", qids, PROMPTS, out, max_new_tokens=4, batch_size=2, labels=("A", "B"), log=lambda *_: None)
    assert len(calls) == 1 and calls[0] <= 2  # only the batch that lost a row was regenerated
    assert [r["text"] for r in again] == [r["text"] for r in first]
    assert all(math.isfinite(r["lp"][0]) for r in again)
    assert {r["i"] for r in read_jsonl(out)} == set(range(len(PROMPTS)))  # the cut line was skipped, no row was glued to it


def test_span_logprob_stops_at_the_first_newline_or_eos(runner):
    ids = runner.tok.encode("Paris\n\nQuestion: who?", add_special_tokens=False)
    lps = [-1.0] * len(ids)
    pieces = [runner.tok.decode([i]) for i in ids]
    first_newline = next(k for k, piece in enumerate(pieces) if "\n" in piece)
    assert runner.span_logprob(ids, lps) == (-1.0 * first_newline, first_newline)
    assert runner.span_logprob([151645] + ids, [-9.0] + lps) == (0.0, 0)  # EOS first: nothing committed
    rows = runner.generate_batch(PROMPTS[:2], 5)
    for row in rows:
        assert row["span_n"] <= row["n_new"] and row["span_lp"] <= 0


def test_content_step_skips_a_leading_whitespace_token():
    blank = {29473}.__contains__  # Mistral's "▁"
    assert content_step([1098, 5], blank) == 0  # "▁A": already content
    assert content_step([29473, 28740], blank) == 1  # "▁" then "1": read the label at step 1
    assert content_step([29473], blank) == 0  # nothing after the marker: stay at step 0
    assert content_step([], blank) == 0


def test_label_token_ids_skip_the_space_marker():
    mistral = pytest.importorskip("transformers").AutoTokenizer
    try:
        tok = mistral.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", local_files_only=True)
    except Exception:
        pytest.skip("Mistral tokenizer is not cached")
    ids = [label_token_ids(tok, d) for d in "1234"]
    assert all(len(i) == 1 for i in ids) and len({i[0] for i in ids}) == 4  # one distinct digit token each, not the shared "▁"
    assert all(label_token_ids(tok, lab) for lab in ("A", "Yes", "No", "CERTAIN", "NO IDEA"))
