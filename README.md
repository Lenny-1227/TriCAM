# TriCAM

This repository contains the original implementation of **TriCAM**, a multimodal recommendation model that combines user--item graph propagation with visual and textual item features. The implementation includes cross-attention, MLP-based prediction, contrastive learning, and MMD-based modality alignment components.

Raw datasets, generated interaction splits, modality features, and trained checkpoints are not redistributed in this repository. A data-splitting script and the corresponding split protocol are provided.

## Repository structure

```text
TriCAM/
├── README.md
├── requirements.txt
└── codes/
    ├── main.py
    ├── Models.py
    ├── utility/
    │   ├── batch_test.py
    │   ├── load_data.py
    │   ├── metrics.py
    │   └── parser.py
    └── data_preprocessing/
        └── prepare_splits.py
```

`main.py`, `Models.py`, and all files under `utility/` are the original TriCAM source files. `prepare_splits.py` contains the dataset-splitting procedures used to prepare the JSON files read by TriCAM.

## Requirements

The original code requires an NVIDIA GPU. CPU-only execution is not supported because tensors and model components are moved to CUDA directly.

Recommended environment:

```text
Python 3.8
PyTorch 1.10.2 + CUDA 11.3
PyTorch Geometric 2.0.3
gensim 3.8.3
sentence-transformers 2.2.0
```

Install a CUDA-enabled build of PyTorch and the matching PyTorch Geometric packages first. Then install the remaining packages:

```bash
pip install -r requirements.txt
```

The code imports `torch-scatter`, `torch-sparse`, `torch-cluster`, and `torch-spline-conv`; install builds compatible with the selected PyTorch and CUDA versions.

## Datasets

The experiments use the following datasets:

- `MenClothing`
- `WomenClothing`
- `Beauty`
- `MicroLens`

The raw datasets must be downloaded from their original providers:

- Men Clothing and Women Clothing: the data format follows the [MAML repository](https://github.com/liufancs/MAML).
- Beauty: download the Beauty 5-core reviews, metadata, and image features from the [UCSD Amazon Review Data](https://mcauleylab.ucsd.edu/public_datasets/data/amazon/datasets.html).
- MicroLens: download the interaction data and modality features from the [official MicroLens repository](https://github.com/westlake-repl/MicroLens).

The expected processed layout is:

```text
codes/
├── data/
│   ├── MenClothing/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── image_feat.npy
│   │   ├── text_feat.npy
│   │   └── 5-core/{train,val,test}.json
│   ├── WomenClothing/
│   │   └── ...
│   ├── Beauty/
│   │   ├── meta-data/reviews_Beauty_5.json.gz
│   │   ├── image_feat.npy
│   │   ├── text_feat.npy
│   │   └── 5-core/{train,val,test}.json
│   └── MicroLens/
│       ├── microlens.inter
│       ├── image_feat.npy
│       ├── text_feat.npy
│       └── 5-core/{train,val,test}.json
└── models/
```

The rows of `image_feat.npy` and `text_feat.npy` must follow the item-ID order used by the JSON files. Each split file maps a zero-based user ID to a list of zero-based item IDs:

```json
{
  "0": [12, 28, 41],
  "1": [3, 19]
}
```

## Preparing the data splits

The training program does not split data automatically. It directly reads `train.json`, `val.json`, and `test.json` from the selected dataset's `5-core/` directory.

After downloading the raw files into `codes/data/`, run the following commands from `codes/`:

```bash
python data_preprocessing/prepare_splits.py --dataset MenClothing
python data_preprocessing/prepare_splits.py --dataset WomenClothing
python data_preprocessing/prepare_splits.py --dataset Beauty
python data_preprocessing/prepare_splits.py --dataset MicroLens
```

Each command processes only the dataset named by `--dataset`. The other dataset directories do not need to exist. For example, a user who only wants Men Clothing needs only `codes/data/MenClothing/` and should run:

```bash
python data_preprocessing/prepare_splits.py --dataset MenClothing
```

The default random seed is `123`. It can be supplied explicitly:

```bash
python data_preprocessing/prepare_splits.py --dataset Beauty --seed 123
```

### Split protocols

For Men Clothing and Women Clothing, interactions from `train.csv` and `test.csv` are combined. For each user, two interactions are held out if the user has fewer than 10 interactions; otherwise approximately 20% are held out. Half of the held-out interactions are assigned to validation and half to testing.

Beauty first converts the original user and item identifiers to integer IDs and then uses the same per-user random split protocol.

MicroLens already stores the split assignment in the `x_label` column of `microlens.inter`:

```text
x_label = 0  training
x_label = 1  validation
x_label = 2  testing
```

## Modality features

TriCAM additionally requires `image_feat.npy` and `text_feat.npy`; interaction JSON files alone are not sufficient to run the model.

- Men Clothing and Women Clothing: MAML provides image features and Doc2Vec resources in its [processed-data folder](https://drive.google.com/drive/folders/1OuBAC-O9oK5JBgVTUqmSlAsWQ7OPtirr).
- Beauty: the UCSD Amazon data page provides product metadata and image features.
- MicroLens: the official project provides extracted multimodal features.

Prepare or obtain the corresponding feature matrices and save them as `image_feat.npy` and `text_feat.npy` in the selected dataset directory. `prepare_splits.py` only creates the interaction JSON files; it does not create modality features.

Feature matrices must satisfy both conditions below:

1. the number of rows covers every item ID referenced by the JSON files;
2. row `i` contains the visual or textual feature of item ID `i`.

The original training code accepts different input feature dimensions because both modalities are projected to `--feat_embed_dim` by learned linear layers.

## Running TriCAM

Run all training commands from `codes/`, because the original code uses relative data and checkpoint paths:

```bash
cd codes
mkdir -p models
```

Example:

```bash
python main.py \
  --dataset MenClothing \
  --model_name TriCAM \
  --n_layers 2 \
  --alpha 1.0 \
  --beta 0.5 \
  --embed_size 64 \
  --feat_embed_dim 64 \
  --batch_size 1024 \
  --lr 5e-5 \
  --seed 123
```

The model is evaluated on the validation set every `--verbose` epochs. The checkpoint with the best validation NDCG@20 is saved to:

```text
codes/models/<Dataset>_<model_name>
```

After training, the best checkpoint is loaded and evaluated on the test set. Precision, Recall, Hit Ratio, and NDCG are reported at the cutoffs specified by `--Ks` (default: `[10, 20]`).

## Main arguments

| Argument | Default | Description |
| --- | ---: | --- |
| `--dataset` | `MenClothing` | Dataset folder name |
| `--model_name` | none | Checkpoint name; this must be provided |
| `--seed` | `123` | Training random seed |
| `--epoch` | `1000` | Maximum number of epochs |
| `--batch_size` | `1024` | Training batch size |
| `--lr` | `5e-5` | Learning rate |
| `--embed_size` | `64` | Collaborative embedding size |
| `--feat_embed_dim` | `64` | Projected modality feature size |
| `--n_layers` | `1` | Number of graph convolution layers |
| `--alpha` | `1.0` | Weight of the self-node feature |
| `--beta` | `0.5` | Fine-grained interest matching coefficient |
| `--agg` | `concat` | `sum`, `weighted_sum`, `concat`, or `fc` |
| `--Ks` | `[10, 20]` | Ranking evaluation cutoffs |
| `--verbose` | `5` | Validation interval |
| `--early_stopping_patience` | `10` | Early-stopping patience |
| `--use_contrastive` | disabled | Enable contrastive learning |
| `--lambda_mmd` | `0.1` | MMD-loss coefficient |

### Boolean argument behavior

Several original arguments use `action="store_false"`. Their components are enabled by default, and passing the argument disables them:

```text
--use_cross_attention   disables cross-attention
--use_mlp               disables the prediction MLP
--has_norm              disables normalization
--target_aware          disables target-aware scoring
```

In contrast, `--use_contrastive`, `--cf`, and `--lightgcn` are disabled unless explicitly supplied.

## Implementation notes

- Although `--data_path` exists, modality feature loading in `main.py` is hard-coded to `data/<Dataset>/`; use the documented directory layout.
- The loader reads `5-core/` directly. The original `--core` argument does not change this path.
- `--model_name` has no default but is used to construct the checkpoint path, so it must be supplied.
- Dataset files, generated features, checkpoints, logs, and Python caches are excluded by `.gitignore`.
