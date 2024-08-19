from typing import Dict

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer
from wordfreq import word_frequency


def score_word_frequency(word: str) -> float:
    return word_frequency(word, "en")


class Pipeline:
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        top_k: int = 100,
        likelihood_weight: float = 0.5,
        frequency_weight: float = 0.5,
    ):
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForMaskedLM.from_pretrained(model_name)

        self._top_k = top_k
        self._likelihood_weight = likelihood_weight
        self._frequency_weight = frequency_weight

    def mask_token_at_char_index(self, before: str, selection: str, after: str) -> str:
        toks_before = self._tokenizer.tokenize(before)
        toks_selection = self._tokenizer.tokenize(selection)
        toks_after = self._tokenizer.tokenize(after)

        if len(toks_selection) > 1:
            toks_after = toks_selection[1:] + toks_after
            toks_selection = toks_selection[:1]

        toks_before = toks_before[max(0, len(toks_before) - 256) :]
        toks_after = toks_after[: min(len(toks_after), 255)]

        tokens = toks_before + [self._tokenizer.mask_token] + toks_after
        return self._tokenizer.convert_tokens_to_string(tokens)

    def score_word_likelihood_from_context(self, masked_context: str) -> torch.Tensor:
        inputs = self._tokenizer(masked_context, return_tensors="pt", truncation=True)
        mask_token_index = torch.where(inputs["input_ids"] == self._tokenizer.mask_token_id)[1]

        with torch.no_grad():
            outputs = self._model(**inputs)

        mask_token_logits = outputs.logits[0, mask_token_index, :]
        return F.softmax(mask_token_logits, dim=-1)

    def top_scoring_tokens(
        self,
        masked_context: str,
        top_k: int = 100,
        likelihood_weight: float = 0.5,
        frequency_weight: float = 0.5,
    ) -> Dict[int, float]:
        word_likelihoods = self.score_word_likelihood_from_context(masked_context)
        top_probs, top_indices = torch.topk(word_likelihoods, top_k)
        score_probs = top_probs[0]
        scores_freq = torch.tensor([score_word_frequency(self._tokenizer.decode([idx])) for idx in top_indices[0]])
        return dict(
            zip(
                top_indices[0],
                likelihood_weight * score_probs + frequency_weight * scores_freq,
            )
        )

    def predict(self, before: str, selection: str, after: str) -> Dict[str, float]:
        masked_context = self.mask_token_at_char_index(before, selection, after)
        tok_to_score = self.top_scoring_tokens(
            masked_context, self._top_k, self._likelihood_weight, self._frequency_weight
        )
        top_tok_idx = max(tok_to_score, key=lambda k: tok_to_score[k])
        top_tok = self._tokenizer.decode([top_tok_idx])
        return top_tok, tok_to_score[top_tok_idx].item()
