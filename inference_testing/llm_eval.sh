nohup lm_eval --model hf \
    --model_args  pretrained=Qwen/Qwen3-0.6B \
    --tasks hendrycks_math \
    --device cpu \
    --batch_size 8 > lm_eval.log 2>&1 &