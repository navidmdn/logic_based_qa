from nl2log.data_loader import MetaQADataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

"""
checking how many of the predicted logical statements are exactly the same as 
the ground truth logical statements in the test set
"""


base_path = './data'
translator_model_cp = './results/checkpoint-7000'
loader = MetaQADataLoader(base_path)
train_raw_data, test_raw_data, dev_raw_data = loader.load_raw_data(base_path)

translator_tokenizer = AutoTokenizer.from_pretrained(translator_model_cp)
translator_model = AutoModelForSeq2SeqLM.from_pretrained(translator_model_cp)


def get_translated_output(question_list, tokenizer, model, batch_size=1000):
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

        outputs = model.generate(inputs, max_length=200)
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
    return num_correct / len(translated_output)


questions = [x[0] for x in test_raw_data][:1000]
logical_steps = [x[1] for x in test_raw_data][:1000]
translated_output = get_translated_output(questions, translator_tokenizer, translator_model)
print(f"accuracy: {evaluate_translations_exact_accuracy(translated_output, logical_steps)}")