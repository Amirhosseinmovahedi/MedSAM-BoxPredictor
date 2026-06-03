# Data

Place your dataset(s) in this directory. Each dataset should follow this structure:

```
data/
└── <DATASET_NAME>/
    ├── imgs/
    ├── imgs_val/
    ├── gts/
    └── gts_val/
```

- `imgs/` — training images
- `imgs_val/` — validation images
- `gts/` — training ground truth segmentation masks
- `gts_val/` — validation ground truth segmentation masks

All images and masks should be in **`.png`** format (default expected by the code).
