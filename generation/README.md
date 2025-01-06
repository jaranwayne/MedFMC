
## Environment Setup
```bash
cd SDXL-Train

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple some-package

pip install accelerate==0.16.0 -i https://pypi.tuna.tsinghua.edu.cn/simple some-package

pip install torch==2.0.1 -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
```

## Training Workflow
```bash
dataset
├── caption
│   └── image1.caption
├── images
│   └── image1.png
├── json
└── tag
    └── iamge1.txt
```

### Prepare the Dataset
1.	For each image, create corresponding caption and tag files.

2.	Merge the generated .caption and .txt annotation files into a single JSON file. This step ensures that the training data and annotations are efficiently loaded during SDXL model training.
```bash
cd SDXL-Train
python ./finetune/merge_all_to_metadata.py "./dataset" "./dataset/meta_clean.json"
```
3.	After preparing the annotation files, organize the data into buckets and save latent features. Additionally, store image resolution information in a new JSON file, meta_lat.json. Latent features will be saved as .npz files, enabling faster data loading during training.
```bash
cd SDXL-Train
python ./finetune/prepare_buckets_latents.py "./dataset" "./dataset/meta_clean.json" "./dataset/meta_lat.json" "model_path" --batch_size 4 --max_resolution "512,512"
```

### Training Phase
1.	The XL_config folder contains two configuration files: \
	• config_file.toml: Stores hyperparameters for SDXL training.\
	• sample_prompt.toml: Contains validation prompts for training.
```bash
train_config
└── XL_config
    ├── config_file.toml
    └── sample_prompt.toml
```

2.	Set up the accelerate configuration.
3.	Start the training process:
```bash
cd SDXL-Trian

sh SDXL_finetune.sh
```

## Generate Images
```bash
 python gen_imgs.py \
    --model_path "" \
    --csv_path "" \
    --output_dir "" \
    --caption_column "" \
    --output_csv "" \
    --num_images xx \
    --batch_size xx
    --enable_xformers
```