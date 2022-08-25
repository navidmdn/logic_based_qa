python nl2log/trainer.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --do_predict \
    --train_file ./data/train.json \
    --validation_file ./data/dev.json \
    --test_file ./data/test.json \
    --source_prefix "summarize: " \
    --output_dir ./results/t5-v3 \
    --overwrite_output_dir \
    --per_device_train_batch_size=32 \
    --per_device_eval_batch_size=64 \
    --predict_with_generate \
    --evaluation_strategy "steps" \
    --eval_steps 1000 \
    --num_train_epochs 3 \
    --generation_max_length 200
