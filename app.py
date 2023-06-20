from qa.question_answering import QuestionAnswering
from qa.data_loader import MetaQADataLoader
import re
from typing import List


def get_answer_paths(answers: List[str], prolog_results):


def answer_question(model_name_or_path: str, question: str):
    data_loader = MetaQADataLoader('./data')
    qa = QuestionAnswering(model_name_or_path, data_loader)

    question_ent = re.findall(r'\[(.+)\]', question)[0]
    question = question.replace(question_ent, 'ENT')
    predicate = qa.nl2predicates(question)
    predicate = predicate.replace('ENT', question_ent)
    print(predicate)
    results = qa.prolog_da.query_with_trace(predicate, question_ent)
    return results


print(answer_question(
    "navidmadani/nl2logic_t5small_metaqa",
    question="who are the actors in the films written by [John Travis]"
))