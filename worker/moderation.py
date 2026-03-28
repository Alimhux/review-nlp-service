import logging

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

from common.models import ReviewStatus

logger = logging.getLogger(__name__)

# Порог уверенности: если score toxic >= TOXIC_THRESHOLD, отзыв отклоняется.
# 0.5 = стандартный (берём только если модель уверена больше чем на 50%)
# 0.4 = более строгий (ловит больше пограничных случаев)
TOXIC_THRESHOLD = 0.4

# Готовые русскоязычные модели:
# - токсичность: s-nlp/russian_toxicity_classifier
# - спам: RUSpam/spam_deberta_v4
_toxic_pipe = pipeline(
    "text-classification",
    model="s-nlp/russian_toxicity_classifier",
    tokenizer="s-nlp/russian_toxicity_classifier",
    top_k=None,
)

_spam_pipe = pipeline(
    "text-classification",
    model="RUSpam/spam_deberta_v4",
    tokenizer="RUSpam/spam_deberta_v4",
    top_k=None,
)


def _spam_score(text: str) -> tuple[bool, float]:
    """Возвращает (is_spam, score)."""
    scores = _spam_pipe(text, truncation=True)
    scores = scores[0] if isinstance(scores[0], list) else scores
    spam_score = 0.0
    for item in scores:
        if item["label"].lower() in ("spam", "LABEL_1"):
            spam_score = item["score"]
            break
    return spam_score > 0.5, spam_score


def _toxic_score(text: str) -> tuple[bool, float]:
    """Возвращает (is_toxic, toxic_score)."""
    result = _toxic_pipe(text, truncation=True)
    scores = result[0] if isinstance(result[0], list) else result
    # Ищем score именно для метки "toxic"
    toxic_score = 0.0
    for item in scores:
        if item["label"].lower() == "toxic":
            toxic_score = item["score"]
            break
    return toxic_score >= TOXIC_THRESHOLD, toxic_score


def moderate_text(text: str) -> tuple[ReviewStatus, str | None]:
    """Используем готовые модели для токсичности и спама на русском."""
    is_toxic, toxic_sc = _toxic_score(text)
    is_spam, spam_sc = _spam_score(text)

    logger.info("Moderation scores — toxic: %.4f (threshold %.2f), spam: %.4f",
                toxic_sc, TOXIC_THRESHOLD, spam_sc)

    if is_toxic:
        reason = f"Отклонено: токсичный текст (score {toxic_sc:.2f})"
        logger.info("Verdict: REJECTED — %s", reason)
        return ReviewStatus.rejected, reason
    if is_spam:
        reason = f"Отклонено: спам (score {spam_sc:.2f})"
        logger.info("Verdict: REJECTED — %s", reason)
        return ReviewStatus.rejected, reason

    logger.info("Verdict: APPROVED")
    return ReviewStatus.published, None
