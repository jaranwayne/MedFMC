accelerate launch \
  --config_file accelerate_config.yaml \
  --num_cpu_threads_per_process=48 \
  /data/home/wudezhi/collapse/SDXL-Train/sdxl_train.py \
  --sample_prompts="/data/home/wudezhi/collapse/SDXL-Train/train_config/XL_config/sample_prompt.toml" \
  --config_file="/data/home/wudezhi/collapse/SDXL-Train/train_config/XL_config/config_file.toml"