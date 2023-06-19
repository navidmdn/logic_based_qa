from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import Union, List
import re
from tqdm import tqdm


class QuestionAnswering:
    def __init__(self, pretrained_translator_path: str, data_loader, cache_dir=None):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # for accessing prolog module
        self.prolog_da = data_loader.prolog_da
        self.nl2log_model, self.nl2log_tokenizer = self.load_pretrained_translator(pretrained_translator_path, cache_dir)

    def load_pretrained_translator(self, pretrained_translator_path, cache_dir=None):

        translator_tokenizer = AutoTokenizer.from_pretrained(pretrained_translator_path, cache_dir=cache_dir)
        translator_model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_translator_path, cache_dir=cache_dir)
        translator_model.eval()
        translator_model = translator_model.to(self.device)

        return translator_model, translator_tokenizer

    def _nl2log_single(self, question: str) -> str:
        question = f"predicates: {question}"
        inputs = self.nl2log_tokenizer(
            question,
            return_tensors="pt",
            max_length=200,
            truncation=True,
            padding=True
        ).input_ids
        outputs = self.nl2log_model.generate(inputs.to(self.device), max_new_tokens=100)
        return self.nl2log_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _nl2log_batch(self, question: List[str], batch_size: int = 512) -> List[str]:
        question = [f"logical form: {q}" for q in question]
        inputs = self.nl2log_tokenizer(
            question,
            return_tensors="pt",
            max_length=100,
            truncation=True,
            padding=True
        ).input_ids
        results = []
        for i in tqdm(range(0, len(question), batch_size)):
            outputs = self.nl2log_model.generate(inputs[i:i + batch_size].to(self.device), max_new_tokens=100)
            results.extend([self.nl2log_tokenizer.decode(out, skip_special_tokens=True) for out in outputs])
        return results

    def nl2predicates(self, questions: Union[str, List[str]], batch_size: int = 512) -> Union[str, List[str]]:
        if isinstance(questions, str):
            return self._nl2log_single(questions)
        else:
            return self._nl2log_batch(questions, batch_size)

    def answer_question_by_precalculated_predicate(self, question_ent: str, predicate: str):
        try:
            results = self.prolog_da.query(predicate, question_ent)
            return results
        except Exception as e:
            print("EXXXX")
            print(f'qent:{question_ent}, predicate:{predicate}')

            print(e)

    def answer_question(self, question: str) -> str:
        question_ent = re.findall(r'\[(.+)\]', question)[0]
        predicate = self.nl2predicates(question)
        results = self.prolog_da.query(predicate, question_ent)
        return results


