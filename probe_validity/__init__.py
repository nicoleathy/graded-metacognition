"""Validity of verbalized-confidence probes (Paper 1).

Design rules: no prompt contains a response-format instruction inside question content; confidence labels are
checked against answer labels; replies are parsed strictly and invalid ones are never imputed; every direct answer is
generated once per (model, dataset) and shared by all probes; nothing is truncated; every run records its full
generation config. The Hugging Face token is never sent implicitly: gated models need an explicit opt-in.
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

__version__ = "0.1.0"
