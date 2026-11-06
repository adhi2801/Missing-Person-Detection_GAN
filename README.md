# Sketch-to-Photo GAN for Missing-Person Identification

Police sketches of a missing or wanted person are hard to compare with real photos. This project trains a conditional
GAN (pix2pix-style) in Keras that turns a face sketch into a photo-like image, as a first step towards matching
sketches against photo databases. The matching step itself is not implemented here.

## What is in the repo

| Notebook | What it does |
|---|---|
| [`Data Augmentation.ipynb`](Data%20Augmentation.ipynb) | Grows the training set with random rotation, shear and translation, applied identically to each sketch and its photo |
| [`Testing.ipynb`](Testing.ipynb) | Builds and trains the GAN: a U-Net-style generator with skip connections and instance normalisation, and a PatchGAN discriminator. The generator loss is adversarial + 100 × (a weighted mix of pixel L1 loss and contextual loss). 100 epochs, batch size 16 |
| [`Predict Image.ipynb`](Predict%20Image.ipynb) | Loads a trained generator and shows sketch → generated photo → real photo |
| [`Compute SSIM and L2-norm.ipynb`](Compute%20SSIM%20and%20L2-norm.ipynb) | Compares generated photos with the real ones for each loss setting |

Data: the CUHK face sketch database (pairs of face photos and artist sketches). The data and the trained models are
not included.

## Loss-weight experiment

Four generators were trained with different weights between pixel loss and contextual loss:

| Pixel weight | Contextual weight | L2 (as computed) | SSIM (as computed) |
|---|---|---|---|
| 1.0 | 0.0 | 90.68 | 0.786 |
| 0.8 | 0.2 | 92.39 | 0.771 |
| 0.5 | 0.5 | 93.45 | 0.727 |
| 0.2 | 0.8 | 93.63 | 0.766 |

**These numbers are not reliable, and I would not draw conclusions from them.** Looking back at the evaluation
notebook, it has two bugs:

1. `compute_ssim` collects a score per image but returns `np.mean(score)`, which is the score of the **last image
   only**. It should return `np.mean(ssim_scores)`.
2. `compute_l2` subtracts two `uint8` images. Unsigned 8-bit values wrap around (3 − 5 becomes 254), so the "L2"
   values are inflated and noisy. The images should be cast to `float` before subtracting.

Both are one-line fixes, but the generated images and models are not in this repo, so the table above has not been
recomputed. The qualitative examples in `Predict Image.ipynb` are the better guide to what the model does.

## Running it

```bash
pip install -r requirements.txt   # also needs keras-contrib for InstanceNormalization, scikit-image and natsort
jupyter notebook
```

Update the dataset and model paths at the top of each notebook.

## References

- Isola et al., [Image-to-Image Translation with Conditional Adversarial Networks (pix2pix)](https://arxiv.org/abs/1611.07004), 2017
- Mechrez et al., [The Contextual Loss for Image Transformation with Non-Aligned Data](https://arxiv.org/abs/1803.02077), 2018
- Wang & Tang, Face Photo-Sketch Synthesis and Recognition, IEEE TPAMI 2009 (CUHK face sketch database)
