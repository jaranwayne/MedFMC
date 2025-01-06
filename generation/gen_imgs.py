import argparse
import os
import random
import pandas as pd
import logging
from diffusers import StableDiffusionXLPipeline
import torch
from accelerate import Accelerator
import csv

accelerator = Accelerator()
device = accelerator.device

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def parse_args():
    parser = argparse.ArgumentParser(description="Stable Diffusion XL batch image generation.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for image generation.")
    parser.add_argument("--output_dir", type=str, default="./final", help="Directory to save generated images.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file containing prompts.")
    parser.add_argument("--caption_column", type=str, default="prompt", help="Column name for captions in the CSV.")
    parser.add_argument("--width", type=int, default=512, help="Width of the generated images.")
    parser.add_argument("--height", type=int, default=512, help="Height of the generated images.")
    parser.add_argument("--num_images", type=int, default=None, help="Number of images to generate.")
    parser.add_argument("--output_csv", type=str, default="generated_images.csv", help="Output CSV file to log image paths and prompts.")
    parser.add_argument("--enable_xformers", action="store_true", help="Enable xformers memory-efficient attention.")
    return parser.parse_args()

def load_prompts_from_csv(csv_path, caption_column):
    if not os.path.exists(csv_path):
        logging.error(f"CSV file not found: {csv_path}")
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if caption_column not in df.columns:
        logging.error(f"Column '{caption_column}' not found in the CSV file.")
        raise ValueError(f"Column '{caption_column}' not found in the CSV file.")
    return df[caption_column].dropna().tolist()

def get_image_count(output_dir):
    if not os.path.exists(output_dir):
        return 0
    return len([f for f in os.listdir(output_dir) if f.endswith((".png", ".jpg", ".jpeg"))])

def generate_images(pipe, prompts, batch_size, output_dir, width, height, output_csv, num_images=None):
    os.makedirs(output_dir, exist_ok=True)
    current_count = get_image_count(output_dir)

    if not os.path.exists(output_csv):
        with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "prompt"])  

    if num_images is not None:
        prompts = prompts[:num_images]
        
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        
        logging.info(f"Processing batch {i // batch_size + 1}...")

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                images = pipe(prompt=batch_prompts, height=height, width=width).images

        with open(output_csv, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for idx, img in enumerate(images):
                current_count += 1
                img_path = os.path.join(output_dir, f"image_{current_count:06d}.png")
                img.save(img_path)
                logging.info(f"Saved: {img_path}")

                writer.writerow([img_path, batch_prompts[idx]])

                if num_images is not None and current_count >= num_images:
                    logging.info("Reached the target number of images. Stopping generation.")
                    return

    logging.info(f"All images have been saved and logged to: {output_csv}")

def main():
    args = parse_args()
   
    try:
        logging.info(f"Loading model from: {args.model_path}")
        pipe = StableDiffusionXLPipeline.from_single_file(
            args.model_path,
            use_safetensors=True,
            local_files_only=True
        )
        pipe.to(device)
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return

    if args.enable_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logging.info("Xformers memory-efficient attention enabled.")
        except Exception as e:
            logging.warning(f"Failed to enable xformers: {e}")

    try:
        logging.info(f"Loading prompts from CSV: {args.csv_path}")
        prompts = load_prompts_from_csv(args.csv_path, args.caption_column)
        logging.info(f"Loaded {len(prompts)} prompts.")
    except Exception as e:
        logging.error(f"Error loading prompts: {e}")
        return

    random.shuffle(prompts)
    logging.info("Prompts have been shuffled.")

    pipe = accelerator.prepare(pipe)

    logging.info(f"Starting generation with batch_size={args.batch_size}, output_dir={args.output_dir}, num_images={args.num_images}...")
    generate_images(pipe, prompts, args.batch_size, args.output_dir, args.width, args.height, args.output_csv, args.num_images)

    logging.info("All images have been generated and saved.")

if __name__ == "__main__":
    main()