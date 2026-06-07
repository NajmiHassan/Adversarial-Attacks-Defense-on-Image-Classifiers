# AADIC — Adversarial Attacks & Defenses on Image Classification

A single-notebook (`AADIC.ipynb`) experiment that trains an image classifier on
CIFAR-10, breaks it with two standard adversarial attacks, then retrains it with
adversarial training to show that robustness can be recovered at the cost of some
clean accuracy.

The whole project is contained in one notebook and runs end-to-end from top to
bottom.

---

## What the project does (exact pipeline)

The notebook runs three phases in its `__main__` block:

1. **Phase 1 — Standard training:** Train a ResNet-18 on clean CIFAR-10, then
   evaluate it on clean images, FGSM adversarial images, and PGD adversarial images.
2. **Phase 2 — Adversarial training:** Take a fresh ResNet-18, initialize it with
   the Phase-1 weights, retrain it on PGD adversarial examples generated on the fly,
   then evaluate it the same three ways (clean / FGSM / PGD).
3. **Phase 3 — Visualization:** Save side-by-side images of clean vs. FGSM vs. PGD
   examples, and a bar chart comparing all six accuracy numbers.

---

## How each part works

### Configuration
All hyperparameters live in one config cell:

| Setting | Value | Meaning |
|---|---|---|
| `BATCH_SIZE` | 128 | mini-batch size |
| `NUM_EPOCHS` | 15 | epochs for standard training |
| `ADV_EPOCHS` | 10 | epochs for adversarial training |
| `EPS` | 8/255 | perturbation budget (L-infinity norm) |
| `PGD_STEPS` | 10 | number of PGD iterations |
| `PGD_ALPHA` | 2/255 | PGD per-step size |

Device is set to CUDA if available, otherwise CPU.

### Data (`get_cifar10_loaders`)
Downloads CIFAR-10 to `./data`. Training images use random crop (32, padding 4) and
random horizontal flip; test images are left as-is. **No normalization is applied** —
pixel values stay in the `[0, 1]` range, which lets the attacks clamp perturbed
images back into a valid range with a simple `clamp(x, 0, 1)`.

### Model (`get_model`)
A torchvision `resnet18` with random initial weights, adapted for 32×32 input:
- first conv replaced with a 3×3, stride-1 conv (instead of the 7×7 stride-2 used for 224×224 ImageNet images),
- the initial max-pool replaced with `Identity()` so the small feature map isn't downsampled too early,
- final fully-connected layer set to 10 outputs (the 10 CIFAR-10 classes).

### Attacks
- **FGSM** (`fgsm_attack`): a single-step attack. Computes the loss gradient with
  respect to the input and pushes each pixel by `eps` in the direction of the
  gradient's sign: `x_adv = clamp(x + eps * sign(∇x L), 0, 1)`.
- **PGD** (`pgd_attack`): the multi-step version. Starts from a random point inside
  the `eps`-ball, then repeats `PGD_STEPS` times: take a step of size `alpha` along
  the gradient sign, then project back so the perturbation stays within `±eps` of the
  original image and inside `[0, 1]`. This is the stronger, "standard" attack used to
  judge robustness.

### Training
- **`train_standard`**: ordinary supervised training on clean images. SGD
  (lr 0.1, momentum 0.9, weight decay 5e-4) with cosine-annealing learning-rate schedule and cross-entropy loss.
- **`train_adversarial`**: for every batch it first generates PGD adversarial
  examples from the current model, then trains the model on those adversarial images
  (SGD, lr 0.01). This is Madry-style adversarial training.

### Evaluation (`evaluate`)
Runs the model over the test set and reports accuracy. If an attack function is
passed in, each batch is perturbed by that attack before being classified, so the
same function reports clean, FGSM, or PGD accuracy depending on what's passed.

### Visualization
- `visualize_adversarial_examples`: takes a few test images, generates their FGSM and
  PGD versions, and plots three rows (clean / FGSM / PGD) → saved as
  `adversarial_examples.png`.
- `plot_results`: bar chart of all six accuracy values → saved as
  `robustness_comparison.png`.

---

## How to run

Requirements: `torch`, `torchvision`, `matplotlib`, `numpy`. A GPU is recommended
(PGD adversarial training is slow on CPU).

Open `AADIC.ipynb` and run all cells top to bottom. CIFAR-10 downloads
automatically on first run. The notebook prints per-epoch progress and a final
summary, and writes `adversarial_examples.png` and `robustness_comparison.png` to the
working directory.

---

## Results from the included run

| Model | Clean | FGSM | PGD |
|---|---|---|---|
| Standard (clean-trained) | 91.1% | 5.7% | 0.0% |
| Adversarially trained | 73.0% | 46.7% | 43.9% |

The standard model is accurate on clean images but collapses under attack
(0% against PGD). Adversarial training gives up some clean accuracy (≈91% → 73%)
in exchange for large gains in robustness (0% → ≈44% against PGD). Exact numbers will
vary slightly between runs.
