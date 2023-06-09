export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="./dog"
export OUTPUT_DIR="DreamBoothDerek"
export HF_TOKEN="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"
huggingface-cli login --token $HF_TOKEN

accelerate config default;

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_text_encoder \
  --instance_prompt="a photo of sks person" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 --gradient_checkpointing \
  --use_8bit_adam \
  --set_grads_to_none \
  --enable_xformers_memory_efficient_attention \
  --learning_rate=5e-8 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 \
  --push_to_hub
