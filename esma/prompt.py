# === Original binary prompts ===

DIRECT_QA_PROMPT = """Answer the following question with keywords.
Question: {question}
"""

META_QA_PROMPT = """Do you know the answer to the following question? If you know and are sure about the answer, just return "Yes". If you don't know the answer or are uncertain, just return "No".
Question: {question}
"""  # noqa: E501

DIRECT_QA_KO_PROMPT = """다음 질문에 대해 키워드로 답변해 주세요.
질문: {question}
"""

META_QA_KO_PROMPT = """다음 질문에 대한 답을 알고 있나요? 답을 알고 있고 확실하다면 "예"라고만 답변해 주세요. 답을 모르거나 확실하지 않다면 "아니요"라고만 답변해 주세요.
질문: {question}
"""  # noqa: E501

DIRECT_QA_CN_PROMPT = """请用关键词回答以下问题。
问题: {question}
"""

META_QA_CN_PROMPT = """你知道以下问题的答案吗？如果你知道并确定答案，请仅回答"是"。如果你不知道或不确定，请仅回答"否"。
问题: {question}
"""

DIRECT_QA_ES_PROMPT = """Responde a la siguiente pregunta con palabras clave.
Pregunta: {question}
"""

META_QA_ES_PROMPT = """¿Conoces la respuesta a la siguiente pregunta? Si la conoces y estás seguro de ella, responde únicamente "Sí". Si no la conoces o no estás seguro, responde únicamente "No".
Pregunta: {question}
"""  # noqa: E501

DIRECT_QA_WITH_IDW_PROMPT = """Answer the following question with keywords. If you don't know the answer, just return "I don't know".
Question: {question}
"""  # noqa: E501


# === NEW: Graded metacognition prompts ===

# Graded confidence scale (ABCD) - 4 levels
# NOTE: Question placed first so the model processes it before rating.
GRADED_META_QA_PROMPT = """Question: {question}

How confident are you that you can correctly answer this question? Return ONLY the letter.
A) I am certain I know the answer
B) I think I probably know
C) I'm not sure
D) I definitely do not know
"""  # noqa: E501

# Feeling-of-Knowing (FOK) probe - prospective judgment before answering
# Grounded in Hart (1965) FOK paradigm
# NOTE: Question placed first to ground the model before rating.
# Scale descriptions kept minimal to avoid center-bias in small models.
FOK_META_QA_PROMPT = """Question: {question}

Do you know the answer to this question? Rate your feeling-of-knowing from 1 to 5. Return ONLY the number, nothing else.
1 = I have no idea
2 = I probably don't know
3 = I might know
4 = I think I know
5 = I am sure I know
"""  # noqa: E501

# Judgment-of-Learning (JOL) probe - retrospective judgment after answering
# Requires the model's own answer to be inserted
JOL_META_QA_PROMPT = """You were asked the following question and gave the answer shown below.
How confident are you that your answer is correct?
Rate from 1 to 5, and return only the number.
1 = Almost certainly wrong
2 = Probably wrong
3 = Unsure, could go either way
4 = Probably correct
5 = Almost certainly correct

Question: {question}
Your answer: {answer}
"""  # noqa: E501

# Numeric confidence prompt (alternative to ABCD for finer granularity)
NUMERIC_META_QA_PROMPT = """On a scale from 1 to 10, how confident are you that you can correctly answer the following question? Return only the number.
1 = No idea at all
5 = Could go either way
10 = Absolutely certain

Question: {question}
"""  # noqa: E501

# Map of meta prompt types to their prompt strings
META_PROMPT_TYPES = {
    "binary": META_QA_PROMPT,
    "graded": GRADED_META_QA_PROMPT,
    "fok": FOK_META_QA_PROMPT,
    "jol": JOL_META_QA_PROMPT,
    "numeric": NUMERIC_META_QA_PROMPT,
}