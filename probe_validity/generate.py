"""Frozen generation: explicit greedy config, no truncation, raw logits, fixed batches, resumable JSONL output.

Everything that can change a model's reply is set here and recorded in the run metadata: the chat template, the
tokenisation (the template already contains special tokens, so none are added), the padding side, and the full
generation config (the Hub's sampling defaults and repetition penalty are reset, so greedy really is argmax of the
raw logits and the reported log-probabilities are those of the decoder).
"""

import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from .provenance import sha256_of  # noqa: F401  (re-exported for the experiment scripts)

SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_p", "typical_p", "epsilon_cutoff", "eta_cutoff")


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def label_token_ids(tok, label: str) -> list[int]:
    """Token ids that can carry the label as the first content token of a reply (with or without a leading space).
    Leading whitespace-only tokens are skipped: sentencepiece writes "1" as "▁" then "1"."""
    ids = set()
    for variant in (label, " " + label):
        for tid in tok.encode(variant, add_special_tokens=False):
            piece = tok.decode([tid]).strip()
            if not piece:
                continue
            if label.lower().startswith(piece.lower()):
                ids.add(tid)
            break
    return sorted(ids)


def content_step(ids: list[int], is_blank, limit: int = 2) -> int:
    """Index of the first generated token that is not whitespace-only (at most limit - 1): the position at which a reply
    starts to commit to a label. Mistral's tokenizer emits "▁" before a digit, so its labels are read at step 1."""
    for step, tid in enumerate(ids[: limit - 1]):
        if not is_blank(tid):
            return step
    return min(limit - 1, max(len(ids) - 1, 0)) if ids and is_blank(ids[0]) else 0


class Runner:
    def __init__(self, model_id: str, device: str | None = None, dtype: str = "bfloat16", local_only: bool = True,
                 attn: str | None = None, model=None, tok=None):
        self.model_id = model_id
        self.device = device or pick_device()
        self.dtype = dtype
        # SDPA returns NaN for left-padded rows on Apple MPS (any dtype); eager attention does not.
        self.attn = attn or ("eager" if self.device == "mps" else "sdpa")
        self.tok = tok or AutoTokenizer.from_pretrained(model_id, padding_side="left", local_files_only=local_only)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype), local_files_only=local_only,
                                                         attn_implementation=self.attn)
        self.model = model.to(self.device).eval()
        self.freeze_defaults()
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else {self.tok.eos_token_id}
        self.max_positions = getattr(self.model.config, "max_position_embeddings", None)

    def render(self, user_text: str) -> str:
        if self.tok.chat_template is None:
            return user_text
        msgs = [{"role": "user", "content": user_text}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def freeze_defaults(self) -> None:
        """Reset the Hub's sampling defaults on the model's own generation config. A config object passed to generate()
        is not enough: transformers replaces every field of it that equals the library default (do_sample=False,
        repetition_penalty=1.0) with the model's Hub value, so Qwen2.5 would silently sample with a 1.05 penalty."""
        gc = self.model.generation_config
        for field in SAMPLING_FIELDS:
            setattr(gc, field, None)
        gc.do_sample, gc.num_beams, gc.repetition_penalty = False, 1, 1.0
        gc.pad_token_id = self.tok.pad_token_id

    def gen_kwargs(self, max_new_tokens: int, sampling: dict | None = None) -> dict:
        """The decoding settings, passed to every generate() call as arguments (they override the model's config)."""
        kwargs = {field: None for field in SAMPLING_FIELDS}
        kwargs.update(do_sample=False, num_beams=1, repetition_penalty=1.0, max_new_tokens=max_new_tokens,
                      pad_token_id=self.tok.pad_token_id)
        if sampling:
            kwargs.update(do_sample=True, **sampling)
        return kwargs

    def effective_config(self, max_new_tokens: int, sampling: dict | None = None) -> dict:
        gc = copy.deepcopy(self.model.generation_config)
        gc.update(**self.gen_kwargs(max_new_tokens, sampling))
        return gc.to_dict()

    def meta(self, max_new_tokens: int, sampling: dict | None = None) -> dict:
        probe = self.render("probe")
        return {
            "model_id": self.model_id,
            "revision": getattr(self.model.config, "_commit_hash", None),
            "device": self.device,
            "dtype": self.dtype,
            "attention": self.attn,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": sys.version.split()[0],
            "generation_config": self.effective_config(max_new_tokens, sampling),
            "chat_template_sha256": hashlib.sha256((self.tok.chat_template or "").encode()).hexdigest(),
            "rendered_example": probe,
            "padding_side": self.tok.padding_side,
            "add_special_tokens": False,
            "truncation": False,
        }

    def span_logprob(self, ids: list[int], lps: list[float]) -> tuple[float, int]:
        """Sum of the log-probabilities of the reply's first line (up to a newline or the end of the reply) and its
        token count: the model's own likelihood of the answer it committed to."""
        total, count = 0.0, 0
        for tid, lp in zip(ids, lps):
            if tid in self.eos_ids or "\n" in self.tok.decode([tid]):
                break
            total, count = total + lp, count + 1
        return round(total, 4), count

    def plan(self, prompts: list[str], batch_size: int) -> list[list[int]]:
        """Fixed batches: prompts sorted by token length (longest first, ties by index), cut into groups."""
        lengths = [len(self.tok.encode(self.render(p), add_special_tokens=False)) for p in prompts]
        if self.max_positions and max(lengths, default=0) > self.max_positions - 64:
            raise ValueError(f"prompt of {max(lengths)} tokens does not fit in {self.max_positions} positions")
        order = sorted(range(len(prompts)), key=lambda i: (-lengths[i], i))
        return [order[k : k + batch_size] for k in range(0, len(order), batch_size)]

    @torch.no_grad()
    def generate_batch(self, prompts: list[str], max_new_tokens: int = 32, labels: tuple[str, ...] = (),
                       sampling: dict | None = None, seed: int | None = None) -> list[dict]:
        enc = self.tok([self.render(p) for p in prompts], return_tensors="pt", padding=True, truncation=False,
                       add_special_tokens=False).to(self.device)
        if seed is not None:
            torch.manual_seed(seed)
        out = self.model.generate(**enc, **self.gen_kwargs(max_new_tokens, sampling),
                                  return_dict_in_generate=True, output_logits=True)
        new = out.sequences[:, enc["input_ids"].shape[1]:]
        steps = len(out.logits)
        if any(torch.isnan(step).any() for step in out.logits):
            raise RuntimeError(f"NaN logits from {self.model_id} on {self.device} with {self.attn} attention; "
                               "a NaN row would otherwise be read as a reply")
        chosen = torch.stack([torch.log_softmax(out.logits[t].float(), -1).gather(1, new[:, t : t + 1]).squeeze(1)
                              for t in range(steps)], dim=1).cpu()
        blank = lambda tid: not self.tok.decode([tid]).strip()  # noqa: E731
        steps_used = [content_step(new[b].tolist(), blank) if steps > 1 else 0 for b in range(len(prompts))]
        by_step = {k: torch.log_softmax(out.logits[k].float(), -1) for k in set(steps_used)}
        label_lp = {}
        for lab in labels:
            ids = label_token_ids(self.tok, lab)
            if ids:
                label_lp[lab] = [float(torch.logsumexp(by_step[steps_used[b]][b, ids], dim=-1)) for b in range(len(prompts))]
            else:
                label_lp[lab] = [float("-inf")] * len(prompts)
        rows = []
        for b, prompt in enumerate(prompts):
            ids = new[b].tolist()
            stop = next((t for t, tid in enumerate(ids) if tid in self.eos_ids), None)
            n_new = stop + 1 if stop is not None else steps
            text = self.tok.decode(ids[: stop if stop is not None else steps], skip_special_tokens=True)
            lps = [round(x, 4) for x in chosen[b, :n_new].tolist()]
            span_lp, span_n = self.span_logprob(ids[:n_new], lps)
            rows.append({
                "text": text,
                "n_new": n_new,
                "finish": "eos" if stop is not None else "length",
                "ids": ids[:n_new],
                "lp": lps,
                "span_lp": span_lp,
                "span_n": span_n,
                "label_lp": {lab: round(label_lp[lab][b], 4) for lab in labels},
                "label_step": steps_used[b],
                "prompt_tokens": int(enc["attention_mask"][b].sum()),
            })
        if self.device == "mps":
            torch.mps.empty_cache()
        return rows


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if path.exists():
        for line in path.read_text().split("\n"):  # only a newline ends a record: a reply may hold U+0085 or U+2028, which splitlines() would cut in two
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a line cut short by an interrupted write (or the empty tail): its batch is redone
    return rows


def run_condition(runner: Runner, name: str, qids: list[str], prompts: list[str], out_path: Path,
                  max_new_tokens: int = 32, batch_size: int = 16, labels: tuple[str, ...] = (),
                  sampling: dict | None = None, seed: int = 0, log=print) -> list[dict]:
    """Generate every prompt of one condition and append the rows to out_path. Batches are fixed by prompt length, so
    a restart redoes only batches with a missing row and every batch has the same composition as in the first run."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["i"]: r for r in read_jsonl(out_path) if r.get("condition") == name}
    stale = sorted(i for i, r in done.items() if i >= len(prompts) or r.get("qid") != qids[i]
                   or r.get("prompt_sha256") != hashlib.sha256(prompts[i].encode()).hexdigest())
    if stale:  # a resume must continue the same items and prompts: rows from another manifest or template would mix silently
        raise RuntimeError(f"{out_path.name}: {len(stale)} stored rows do not belong to these items or prompts (first: item {stale[0]}); "
                           "refusing to resume on top of them")
    batches = runner.plan(prompts, batch_size)
    t0, fresh = time.time(), 0
    for number, idx in enumerate(batches):
        if all(i in done for i in idx):
            continue
        rows = runner.generate_batch([prompts[i] for i in idx], max_new_tokens, labels, sampling, seed=seed + number)
        with out_path.open("a") as fh:
            if out_path.stat().st_size and not out_path.read_bytes().endswith(b"\n"):
                fh.write("\n")  # an interrupted write left a cut line: keep the new rows off it
            for i, row in zip(idx, rows):
                row.update(i=i, qid=qids[i], condition=name, prompt_sha256=hashlib.sha256(prompts[i].encode()).hexdigest())
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                done[i] = row
            fh.flush()
        fresh += len(idx)
        if (number + 1) % 5 == 0 or number == len(batches) - 1:
            rate = fresh / max(time.time() - t0, 1e-9)
            log(f"  {name}: {len(done)}/{len(prompts)} done, {rate:.2f} prompts/s")
    return [done[i] for i in range(len(prompts))]
