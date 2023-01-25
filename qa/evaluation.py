from qa.data_loader import MetaQADataLoader
from qa.question_answering import QuestionAnswering
from typing import List
import argparse


def exact_match_score(predictions: List[str], ground_truths: List[str]) -> float:
    assert len(predictions) == len(ground_truths)
    return sum([1 if p == g else 0 for p, g in zip(predictions, ground_truths)]) / len(predictions)


def evaluate_qa_model(nl2log_model_path):
    data_loader = MetaQADataLoader('./data', split='test')
    qa = QuestionAnswering(nl2log_model_path)

    scores = []
    ds_sizes = []
    for hop_name, dataset in data_loader.dataset.items():
        ds_questions, ds_answers = list(zip(*dataset))
        ds_sizes.append(len(ds_questions))
        ds_answers = qa.answer(ds_questions, batch_size=512)
        score = exact_match_score(ds_answers, ds_answers)
        scores.append(score)
        print(f"{hop_name} test accuracy: {exact_match_score(ds_answers, ds_answers)}")

    print(f"Average test accuracy: {sum(scores) / sum(ds_sizes)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/nl2log')
    args = parser.parse_args()
    evaluate_qa_model(args.model_path)


