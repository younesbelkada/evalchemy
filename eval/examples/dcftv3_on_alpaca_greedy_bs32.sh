accelerate launch --num-processes 1 --num-machines 1 -m eval.eval \
    --model hf \
    --task alpaca_eval \
    --model_args 'pretrained=mlfoundations-dev/oh-dcft-v3-sharegpt-format-sedrick' \
    --batch_size 32 \
    --output_path logs \
    --annotator gpt-4-1106-preview-greedy
    
