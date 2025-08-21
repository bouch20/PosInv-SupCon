# Reproducing Experiments on CUB-200-2011

This repository contains code to train and evaluate **Cross-Entropy (CE)**, **PosInv-SupCon**, and **PosInv-SupCon + CE (Joint)** models on the CUB-200-2011 dataset.

* Dataset: **CUB-200-2011**
* Default batch size: **16**
* Reported statistics: **Top-1 accuracy**, averaged over **n=3** runs with seeds **42, 43, 44**

## Results

**Classification Top-1 accuracy on CUB-200-2011** (batch size = 16; seeds = 42, 43, 44).

| Loss Function                        | **Mean ± SD**    | **Range**       |
| ------------------------------------ | ---------------- | --------------- |
| Cross-Entropy                        | 77.22 ± 1.54     | 75.11–78.72     |
| PosInv-SupCon                        | 76.59 ± 0.17     | 76.35–76.75     |
| PosInv-SupCon + CE (Joint λ=1)       | 75.31 ± 0.14     | 75.15–75.49     |
| PosInv-SupCon + CE (Joint λ=0.5)     | 77.30 ± 0.35     | 76.86–77.72     |
| **PosInv-SupCon + CE (Joint λ=0.1)** | **78.91 ± 0.14** | **78.72–79.06** |

> Notes:
>
> * “Joint λ” denotes the weight on the CE term in the joint training objective.
> * Each number is the mean ± standard deviation across the three random seeds listed above.

---

## How to Run — Google Colab

Open the corresponding notebook in **Google Colab**, copy it to your Drive, set any paths if needed, and run all cells.

* **Cross Entropy**
  `notebooks/CUB_200_2011_CE_ResNet_pretrained_ipynb.ipynb`

* **PosInv-SupCon (2-stage model)**
  `notebooks/CUB-200-2011_PosInv-SupCon_ResNet_pretrained_2stage.ipynb`

* **PosInv-SupCon + CE (Joint model)**
  `notebooks/CUB_200_2011_PosInv_SupCon_ResNet_pretrained_JOINT.ipynb`
  *(Set the joint weight λ inside the notebook as needed, e.g., 1, 0.5, or 0.1.)*

---

## How to Run — Local

Make sure your environment satisfies the requirements used by the notebooks/scripts (PyTorch, torchvision, etc.), and that the dataset path is correctly configured wherever the code expects it.

### Cross Entropy

```bash
python train_ce.py --batch_size {batch_size} --model resnet18_pretrain --cosine --warm --seed 42
```

> Repeat with seeds **42, 43, 44** to reproduce the reported mean/SD.

### PosInv-SupCon (2-Stage)

```bash
# 1) Stage 1: contrastive pretraining
python train_posinv_supcon_1stage.py --batch_size {batch_size} --model resnet18_pretrain --cosine --warm

# The best checkpoint path is printed at the end of Stage 1.
# 2) Stage 2: supervised fine-tuning from the best Stage-1 checkpoint
python train_posinv_supcon_2stage.py --batch_size {batch_size} --model resnet18_pretrain --cosine --warm --ckpt {checkpoint_path}
```

### PosInv-SupCon + CE (Joint)

Use the Colab notebook:

```
notebooks/CUB_200_2011_PosInv_SupCon_ResNet_pretrained_JOINT.ipynb
```

Set **λ** inside the notebook to match the configurations in the results table (e.g., 1, 0.5, 0.1).

---

## Reproducibility

* **Seeds:** 42, 43, 44 (as used in the results table).
* **Batch size:** 16 for reported numbers.
* When reproducing results, run each configuration with all three seeds and report **mean ± standard deviation** as above.
