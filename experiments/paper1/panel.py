"""The model panel of Paper 1: which models are run, in which order, and which of them need a licence.

The panel is the eleven models of the earlier draft (five families) plus Llama-3.1-8B, which was already running. Qwen2.5-14B is a
validity-only run (MMLU and the SuperGPQA slice) that tests whether the failures persist at a larger scale. A gated model (Llama,
Gemma) is queued only when its weights are already in the local Hugging Face cache: downloading one needs the owner's licence
acceptance and token, and nothing here reads or uses a token. The runner loads models from the cache only (local_files_only), so a public
model must be downloaded first (download_panel.py) before it can run.
"""

import json
from dataclasses import dataclass
from pathlib import Path

CACHE = Path.home() / ".cache/huggingface/hub"
DATASET_ORDER = ("mmlu", "triviaqa", "mmlu_pro_eng", "supergpqa_mfg", "webquestions")  # the MMLU decomposition, then the open-QA positive control
VALIDITY_ONLY_DATASETS = ("mmlu", "supergpqa_mfg")


@dataclass(frozen=True)
class PanelModel:
    id: str
    short: str  # the name in tables
    family: str
    size_b: float  # nominal size, for ordering rows within a family
    gated: bool  # weights need a licence click-through and a token to download
    role: str = "core"  # core: every dataset; validity: VALIDITY_ONLY_DATASETS only
    queue_rank: int = 0  # position in the panel queue; 0 = already running in the first queue

    @property
    def cache_dir(self) -> Path:
        return CACHE / ("models--" + self.id.replace("/", "--"))

    @property
    def cached(self) -> bool:
        """The cache holds a complete copy of the weights (the config and every safetensors shard), not just a partial download."""
        for snap in sorted((self.cache_dir / "snapshots").glob("*")) if self.cache_dir.exists() else []:
            if not (snap / "config.json").exists():
                continue
            index = snap / "model.safetensors.index.json"
            if index.exists():
                if all((snap / shard).exists() for shard in set(json.loads(index.read_text())["weight_map"].values())):
                    return True
            elif (snap / "model.safetensors").exists():
                return True
        return False

    @property
    def runnable(self) -> bool:
        """Can start now: the runner reads the local cache only."""
        return self.cached

    @property
    def downloadable(self) -> bool:
        """Public weights that are not cached yet: download_panel.py fetches them without a token."""
        return not self.gated and not self.cached

    @property
    def datasets(self) -> tuple[str, ...]:
        return DATASET_ORDER if self.role == "core" else VALIDITY_ONLY_DATASETS


PANEL = (
    PanelModel("Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B", "Qwen2.5", 1.5, False),
    PanelModel("Qwen/Qwen2.5-3B-Instruct", "Qwen2.5-3B", "Qwen2.5", 3.0, False, queue_rank=2),
    PanelModel("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B", "Qwen2.5", 7.0, False),
    PanelModel("Qwen/Qwen2.5-14B-Instruct", "Qwen2.5-14B", "Qwen2.5", 14.0, False, role="validity", queue_rank=9),
    PanelModel("Qwen/Qwen3-1.7B", "Qwen3-1.7B", "Qwen3", 1.7, False, queue_rank=4),
    PanelModel("Qwen/Qwen3-4B", "Qwen3-4B", "Qwen3", 4.0, False, queue_rank=1),  # the one model whose Yes/No probe looked usable on MMLU in the earlier draft
    PanelModel("Qwen/Qwen3-8B", "Qwen3-8B", "Qwen3", 8.0, False, queue_rank=3),
    PanelModel("meta-llama/Llama-3.2-1B-Instruct", "Llama-3.2-1B", "Llama", 1.0, True, queue_rank=6),
    PanelModel("meta-llama/Llama-3.2-3B-Instruct", "Llama-3.2-3B", "Llama", 3.0, True, queue_rank=5),
    PanelModel("meta-llama/Meta-Llama-3.1-8B-Instruct", "Llama-3.1-8B", "Llama", 8.0, True),
    PanelModel("mistralai/Mistral-7B-Instruct-v0.3", "Mistral-7B-v0.3", "Mistral", 7.0, True),
    PanelModel("google/gemma-2-2b-it", "Gemma-2-2B", "Gemma-2", 2.0, True, queue_rank=8),
    PanelModel("google/gemma-2-9b-it", "Gemma-2-9B", "Gemma-2", 9.0, True, queue_rank=7),
)

BY_ID = {m.id: m for m in PANEL}
FAMILY_ORDER = ("Qwen2.5", "Qwen3", "Llama", "Mistral", "Gemma-2")


def short_name(model_id: str) -> str:
    model_id = model_id.replace("__", "/")
    return BY_ID[model_id].short if model_id in BY_ID else model_id.split("/")[-1]


def table_order(model_ids) -> list[str]:
    """Model ids in table order: by family, then size. Ids outside the panel go last, alphabetically."""
    def key(mid):
        m = BY_ID.get(mid.replace("__", "/"))
        return (FAMILY_ORDER.index(m.family), m.size_b, m.short) if m else (len(FAMILY_ORDER), 0.0, mid)
    return sorted(model_ids, key=key)


def panel_queue() -> list[PanelModel]:
    """The models still to run after the first queue, in order, among those whose weights are cached (public ones after download_panel.py)."""
    return sorted((m for m in PANEL if m.queue_rank and m.runnable and m.role == "core"), key=lambda m: m.queue_rank)


def waiting_for_licence() -> list[PanelModel]:
    return [m for m in PANEL if m.queue_rank and m.gated and not m.cached]


def to_download() -> list[PanelModel]:
    return [m for m in PANEL if m.downloadable]


def validity_queue() -> list[PanelModel]:
    return [m for m in PANEL if m.role == "validity" and m.runnable]


if __name__ == "__main__":
    import sys

    if "--queue" in sys.argv:  # "model-id dataset dataset ..." per line, for run_panel_queue.sh
        for m in panel_queue() + validity_queue():
            print(m.id, *m.datasets)
    else:
        for m in PANEL:
            status = "running first queue" if not m.queue_rank else "queued" if m.cached else "needs download" if m.downloadable else "waiting for licence"
            print(f"{m.short:18s} {m.family:8s} {m.role:9s} {'cached' if m.cached else 'not cached':10s} {status}")
