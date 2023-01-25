from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import Union, List


class QuestionAnswering:
    def __init__(self, pretrained_translator_path):
        self.nl2log_model, self.nl2log_tokenizer = self.load_pretrained_translator(pretrained_translator_path)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def load_pretrained_translator(self, pretrained_translator_path):

        translator_tokenizer = AutoTokenizer.from_pretrained(pretrained_translator_path)
        translator_model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_translator_path)
        translator_model = translator_model.to(self.device)

        return translator_model, translator_tokenizer

    def _answer_single(self, question: str) -> str:
        question = f"summarize: {question}"
        inputs = self.nl2log_tokenizer(
            question,
            return_tensors="pt",
            max_length=200,
            truncation=True,
            padding=True
        ).input_ids
        outputs = self.nl2log_model.generate(inputs.to(self.device), max_new_tokens=100)
        return self.nl2log_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _answer_batch(self, question: List[str], batch_size: int = 512) -> List[str]:
        question = [f"summarize: {q}" for q in question]
        inputs = self.nl2log_tokenizer(
            question,
            return_tensors="pt",
            max_length=200,
            truncation=True,
            padding=True
        ).input_ids
        results = []
        for i in range(0, len(question), batch_size):
            outputs = self.nl2log_model.generate(inputs[i:i + batch_size].to(self.device), max_new_tokens=100)
            results.extend([self.nl2log_tokenizer.decode(out, skip_special_tokens=True) for out in outputs])
        return results

    def answer(self, questions: Union[str, List[str]], batch_size: int = 512) -> Union[str, List[str]]:
        if isinstance(questions, str):
            return self._answer_single(questions)
        else:
            return self._answer_batch(questions, batch_size)
