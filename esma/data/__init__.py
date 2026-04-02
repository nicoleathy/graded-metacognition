from .fictional_qa import load_fictional_qa, load_fictional_qa_meta
from .freebase_qa import load_freebase_qa, load_freebase_qa_meta
from .gsm8k import load_gsm8k, load_gsm8k_meta
from .mkqa import load_mkqa, load_mkqa_meta
from .mmlu import load_mmlu, load_mmlu_meta
from .nq_open import load_nq_open, load_nq_open_meta
from .trivia_qa import load_trivia_qa, load_trivia_qa_meta
from .web_questions import load_web_questions, load_web_questions_meta

META_DATASETS = {
    "trivia_qa": load_trivia_qa_meta,
    "fictional_qa": load_fictional_qa_meta,
    "freebase_qa": load_freebase_qa_meta,
    "nq_open": load_nq_open_meta,
    "web_questions": load_web_questions_meta,
    "mkqa": load_mkqa_meta,
    # New transfer benchmarks
    "gsm8k": load_gsm8k_meta,
    "mmlu": load_mmlu_meta,
}

__all__ = [
    "load_trivia_qa",
    "load_trivia_qa_meta",
    "load_fictional_qa",
    "load_fictional_qa_meta",
    "load_freebase_qa",
    "load_freebase_qa_meta",
    "load_nq_open",
    "load_nq_open_meta",
    "load_web_questions",
    "load_web_questions_meta",
    "load_mkqa",
    "load_mkqa_meta",
    "load_gsm8k",
    "load_gsm8k_meta",
    "load_mmlu",
    "load_mmlu_meta",
    "META_DATASETS",
]
