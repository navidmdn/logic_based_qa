from qa.data_loader import MetaQADataLoader
from qa.question_answering import QuestionAnswering
from typing import List
import argparse
import pickle


def accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    assert len(predictions) == len(ground_truths)
    correct = 0
    for p, g in zip(predictions, ground_truths):
        if p is None or g is None:
            continue
        if set(g).issubset(set(p)):
            correct += 1
    return correct / len(predictions)


def log_wrong_answers(predictions: List[str], ground_truths: List[str], questions: List[str]):
    assert len(predictions) == len(ground_truths)
    for p, g, q in zip(predictions, ground_truths, questions):
        if p is None or not set(g).issubset(set(p)):
            print(f"Question: {q}")
            print(f"Prediction: {p}")
            print(f"Ground truth: {g}")
            print("=================================")


def generate_predicates(nl2log_model_path):
    """
    For faster evaluation we first generate predicates given by nl2log model
    and then load predicate file and query using prolog module
    """
    data_loader = MetaQADataLoader('./data', split='test')
    qa = QuestionAnswering(nl2log_model_path, data_loader)

    for hop_name, dataset in data_loader.dataset.items():
        ds_questions, _ = zip(*dataset)
        predicates = qa.nl2predicates(ds_questions, batch_size=256)

        with open(f"./data/{hop_name}_predicates", 'wb') as f:
            pickle.dump(predicates, f)


def evaluate_qa_model(nl2log_model_path):
    data_loader = MetaQADataLoader('./data', split='test')
    qa = QuestionAnswering(nl2log_model_path, data_loader)

    subset_sizes = []
    subset_acc = []

    for hop_name, dataset in data_loader.dataset.items():
        ds_questions, ds_answers = zip(*dataset)
        ds_answers_char_normalized = [data_loader.kb.normalize_chars(ans_set) for ans_set in ds_answers]

        with open(f"./data/{hop_name}_predicates", 'rb') as f:
            ds_predicates = pickle.load(f)

        model_answers = []

        for q, a, p in zip(ds_questions, ds_answers_char_normalized, ds_predicates):
            model_answer = qa.answer_question_by_precalculated_predicate(q, p)
            model_answers.append(model_answer)

        subset_sizes.append(len(model_answers))
        acc = accuracy(model_answers, ds_answers_char_normalized)
        subset_acc.append(acc)
        print(f"{hop_name} accuracy: {acc}")
        log_wrong_answers(model_answers, ds_answers_char_normalized, ds_questions)

    avg_acc = [s * a for s, a in zip(subset_sizes, subset_acc)]
    avg_acc = sum(avg_acc) / sum(subset_sizes)
    print(f"Average accuracy: {avg_acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/nl2log')
    parser.add_argument('--generate_predicates', action='store_true', default=False)
    args = parser.parse_args()

    if not args.generate_predicates:
        print("Evaluating model using precalculated predicates")
        evaluate_qa_model(args.model_path)
    else:
        print("Generating predicates")
        generate_predicates(args.model_path)


