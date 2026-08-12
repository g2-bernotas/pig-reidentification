# Long-term Tracking of Individual Pigs Through Re-identification

Code, trained weights and analysis scripts accompanying the paper on
long-term, non-invasive visual identification of individual pigs.

### Paper
**_Long-term Tracking of Individual Pigs Through Re-identification_**  
**Authors:** Gytis Bernotas, Mark Hansen, Melvyn Smith, Mhairi Jack, Emma Baxter, Richard B. D’Eath  
**Status:** Under review  
**DOI / arXiv:** Will be added upon publication

### Data
**Dataset DOI:** [10.5281/zenodo.20415925](https://doi.org/10.5281/zenodo.20415925)

### Weights
**Model Weights DOI:** [10.5281/zenodo.20415925](https://doi.org/10.5281/zenodo.20415925)


This project implements an open‑set Re-identification (ReID) system for long‑term tracking of individual commercial pigs using computer vision. As commercial pigs tend to have visually uniform coats and their appearance changes daily due to dirt accumulation, traditional short‑term identification methods fail over extended periods.

To address this, the system uses a ResNet‑50 embedding model trained with a combined triplet + softmax objective, incorporating cross‑day positive sampling to ensure that the network learns identity rather than day‑specific appearance. A custom synthetic dirt (“muck”) augmentation pipeline further prevents the model from overfitting to transient contamination patterns.

The system supports open‑set identification, allowing new pigs to be enrolled directly from their embeddings without retraining. Identity stability over time is maintained using a multi‑anchor voting mechanism, which matches daily cluster centroids against a rolling bank of historical representations.

The framework works across multiple input modalities — Instance Segmentation (IS), Oriented Bounding Boxes (OBB), and Axis‑Aligned Bounding Boxes (AABB) — and includes GradCAM‑based interpretability to reveal the anatomical cues the model relies on when distinguishing individuals.

The repository covers three things:

1. **Training and evaluating** the embedding (`train.py`, `test.py`).
2. **Unsupervised longitudinal tracking and enrolment** from released embeddings
   (`analysis/`) — this reproduces the figures and tables in the paper.
3. **Self-supervised daily fine-tuning** (`daily_finetune.py`), the *tuned*
   variant in the paper, in which the model re-trains on pseudo-labels it
   generated itself.

### Background
The code has been adapted from [CWOA/MetricLearningIdentification](https://github.com/CWOA/MetricLearningIdentification), which accompanied Andrew et al., [_"Visual Identification of Individual Holstein Friesian Cattle via Deep Metric Learning"_](https://www.sciencedirect.com/science/article/pii/S0168169921001514) article.

## Data layout

Download the [dataset](https://zenodo.org/records/20415926) and unpack it so that each identity has one folder per
split:

```
data/OpenSetPigs/
    images/
        train/
            G19_238/  <frames>.png
            G19_241/
            ...
        test/
            G19_238/
            G19_241/
            ...
    splits/
        80-20.json
```

`train/` and `test/` must contain the same identity folders. Image filenames
carry the capture date, which is what makes cross-day positive sampling and the
per-day evaluation possible; both the
`Basler_acA4112-20uc__40331001__20240626_101442824_0562.png` and
`G20_270_20250422_...` conventions are parsed automatically.

### Muck augmentation

The dirt augmentation augments each crop with a pre-generated noise image. The
released dataset includes the noise maps; however, they can be generated:

```bash
python utilities/generate_muck_noise.py --out data/muck_noise --count 200
```

Pass the directory as `--muck_dir` (or export `PIG_MUCK_NOISE_DIR`). If no
directory is configured the augmentation is skipped with a warning, which
corresponds to the no-dirt **(ND)** ablation (the same as `--disable_muck`). Supplying `--seg_dir` (a directory of binary masks or instance segmented images) confines the dirt to the animal instead of the whole crop used 
for axis-aligned bounding box crops as well as for the oriented bounding box crops. 

## Training

The following is an example on how a model can be trained. Refer to `python train.py -h` for every option. 

```bash
python train.py \
    --out_path output/inssegm \
    --dataset_root data/OpenSetPigs \
    --folds_file data/OpenSetPigs/splits/80-20.json \
    --muck_dir data/muck_noise \
    --num_folds 5 --num_epochs 30 --batch_size 64 \
    --model TripletResnetSoftmax \
    --loss_function OnlineTripletSoftmaxLoss \
    --triplet_selection SemihardNegative --triplet_margin 0.5 \
    --use_arcface
```

## Evaluating a checkpoint

KNN identification accuracy on the test split, plus embeddings saved to `.npz`:

```bash
python test.py \
    --model_path output/inssegm/fold_0/best_model_state.pkl \
    --dataset_root data/OpenSetPigs \
    --folds_file data/OpenSetPigs/splits/50-50.json \
    --save_path output/inssegm/fold_0
```

t-SNE visualisation of any embeddings file:

```bash
python utilities/visualise_embeddings.py \
    --embeddings_file output/inssegm/fold_0/test_embeddings.npz
```

## Self-supervised daily fine-tuning

```bash
python daily_finetune.py \
    --source-dir data/open_set_crops \
    --work-dir   results/daily_finetune \
    --base-model output/inssegm/fold_0/best_model_state.pkl \
    --base-model-classes 150 \
    --warm-start-days 1 --epochs 3
```

Ground-truth labels are read only to measure accuracy (and for the day-0 warm start (if required)); training labels come from the model's clustering.
Per-day, per-identity accuracies are reported in `accuracies.csv`; the same per-day
overall accuracy is also written to `tracking_overlay.csv`, ready to drop onto
the tracking figure:

```bash
python analysis/evaluate_tracking.py \
    --config analysis/configs/experiments.example.json \
    --out-dir results/tracking \
    --extra-series results/daily_finetune/tracking_overlay.csv
```

## Citation

```bibtex
@article{bernotas2026pigreid,
  title   = {Long-term Tracking of Individual Pigs Through Re-identification},
  author  = {Bernotas, Gytis and Hansen, Mark and Smith, Melvyn and Jack, Mhairi and Baxter, Emma and D'Eath, Richard B.},
  journal = {Under review},
  year    = {2026},
  note    = {Preprint; DOI will be added upon publication}
}
```

Please also cite the works this code builds on:

```bibtex
@article{andrew2020visual,
  title={Visual Identification of Individual Holstein Friesian Cattle via Deep Metric Learning},
  author={Andrew, William and Gao, Jing and Campbell, Neill and Dowsey, Andrew W and Burghardt, Tilo},
  journal={arXiv preprint arXiv:2006.09205},
  year={2020}
}

@inproceedings{lagunes2019learning,
  title={Learning discriminative embeddings for object recognition on-the-fly},
  author={Lagunes-Fortiz, Miguel and Damen, Dima and Mayol-Cuevas, Walterio},
  booktitle={2019 International Conference on Robotics and Automation (ICRA)},
  pages={2932--2938},
  year={2019},
  organization={IEEE}
}
```

## License

MIT — see [`LICENSE`](LICENSE). The dataset and trained weights are released
separately and may carry their own terms; see their records.
