from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import argparse
import json
import torch

"""
checking how many of the predicted logical statements are exactly the same as 
the ground truth logical statements in the test set
"""


def load_test_data(data_path='data/test.json'):
    questions = []
    trans_qs = []
    with open(data_path, 'r') as json_file:
        json_list = list(json_file)
        for json_str in json_list:
            result = json.loads(json_str)
            questions.append(result['question'])
            trans_qs.append(result['logical_steps'])
    return list(zip(questions, trans_qs))


def get_translated_output(question_list, tokenizer, model, device, batch_size=256):
    question_list = [f"summarize: {q}" for q in question_list]
    results = []
    i = 0
    pbar = tqdm(total=len(question_list))

    while i < len(question_list):
        batch = question_list[i:i + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            max_length=200,
            truncation=True,
            padding=True
        ).input_ids

        outputs = model.generate(inputs.to(device), max_new_tokens=100)
        curr_res = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]

        i += batch_size
        pbar.update(batch_size)
        results.extend(curr_res)
    pbar.close()
    return results


def evaluate_translations_exact_accuracy(translated_output, logical_steps):
    num_correct = 0
    for translated, logical_step in zip(translated_output, logical_steps):
        if translated == logical_step:
            num_correct += 1
        else:
            print(translated, logical_step)
    return num_correct / len(translated_output)


def main(test_data_path, translator_model_cp):

    test_ds = load_test_data(test_data_path)

    translator_tokenizer = AutoTokenizer.from_pretrained(translator_model_cp)
    translator_model = AutoModelForSeq2SeqLM.from_pretrained(translator_model_cp)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    translator_model = translator_model.to(device)
    translator_model.eval()

    questions = [x[0] for x in test_ds]
    logical_steps = [x[1] for x in test_ds]
    translated_output = get_translated_output(questions, translator_tokenizer, translator_model, device)
    print(f"exact accuracy: {evaluate_translations_exact_accuracy(translated_output, logical_steps)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path", type=str, default="data/test.json", help="path to the translation"
                                                                                     " ground truth")
    parser.add_argument("--model_cp", type=str, default="results/t5-v3", help="path to the model checkpoint")
    args = parser.parse_args()
    main(args.test_data_path, args.model_cp)
