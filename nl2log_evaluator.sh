python nl2log/trainer.py \
    --model_name_or_path results/ \
    --do_predict \
    --test_file ./data/test.json \
    --source_prefix "summarize: " \
    --output_dir ./results2 \
    --overwrite_output_dir \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=32 \
    --predict_with_generate \
    --evaluation_strategy "steps" \
    --eval_steps 500 \
    --num_train_epochs 1