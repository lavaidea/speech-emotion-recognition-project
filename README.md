# Speech Emotion Recognition Beyond Static Acoustic Features

**Course:** Machine Learning mini-project · **Dataset:** RAVDESS (acted speech, 24 actors, 1,440 speech recordings)

## Research question
How well do handcrafted acoustic features generalize to **unseen speakers** in eight-class speech emotion recognition, and how do temporal dynamics and silence-trimming choices affect performance?

### Hypotheses
- **H1:** Adding temporal summary features (delta/delta-delta MFCC and energy dynamics) improves speaker-grouped macro-F1 over static features.
- **H2:** The silence-trimming threshold changes emotion recognition performance.
- **H3:** Results differ between grouped cross-validation and held-out actors; some emotions are consistently confused.

These are **questions to test**, not guaranteed findings. Interpret SHAP/LIME as explanations of model behavior, not causal evidence about emotion.

## Experimental protocol
- **Train / grouped CV:** actors 01–12; grouping by actor to prevent speaker leakage.
- **Validation:** actors 13–18; fixed speaker-disjoint model comparison.
- **Historical test:** actors 19–24. **Important caveat:** these actors' test results were inspected before the later V2 experiments. Therefore the V2 analysis cannot honestly present this as a pristine, untouched final test. Prefer training-speaker grouped CV and fixed validation for the V2 claims; any subsequent test comparisons must be explicitly labeled exploratory/historical.
- **Primary metric:** macro-F1. Also report balanced accuracy, per-class precision/recall/F1, confusion matrices, and fold-level variability.
- **Baselines:** simple linear models, including logistic regression, compared with SVM and tree ensembles.
- **Leakage control:** fit scalers, PCA, feature selection, and hyperparameter search **inside training folds** using sklearn pipelines. Never mix actors across folds.

## Reproduce in Google Colab
1. Download the RAVDESS **speech audio** from its official distribution, complying with the dataset license. Arrange source WAVs in the location expected by notebook 01; inspect that notebook's path discovery before running.
2. In Google Drive create **one folder**: `MyDrive/speech-emotion-recognition-project/`.
3. Upload the `notebooks/` folder (or open the notebooks in Colab) and run notebooks **01 → 07**, in order. The code uses `PROJECT_DIR = Path('/content/drive/MyDrive/speech-emotion-recognition-project')`.
4. Install packages from `requirements.txt` or the notebook installation cells. Save exported CSV/JSON/model artifacts to the same Drive project folder.


**Note:** Notebook outputs were intentionally cleared in this GitHub package because they were produced in a differently named Drive folder. The code has not been rerun in this environment; all numeric claims must be regenerated and checked in Colab.

## Notebook guide
| Notebook | Purpose |
|---|---|
| 01 | RAVDESS metadata, class balance, speaker structure |
| 02 | Waveforms, Fourier/STFT, mel, F0, silence examples |
| 03 | Static and enriched acoustic feature extraction, trimming variants |
| 04 | Grouped CV, model comparisons, ablations, candidate selection |
| 05 | Validation/test diagnostics, learning curves, errors and limitations |
| 06 | Paired ablation and research conclusions |
| 07 | SHAP/LIME interpretability **(separate Extra Trees analysis; not necessarily the frozen final model)** |


## Dataset and Reproducibility

This project uses the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset.

### Data availability

The dataset and precomputed acoustic features are available in the following Google Drive folder:

[Download RAVDESS project data](https://drive.google.com/drive/folders/1hKh4K0K_2L4z0EoAl1wtNxoENnwth2yV?usp=sharing)

The folder contains:

- `Audio_Speech_Actors_01-24.zip` — original speech recordings.
- `processed/ravdess_features.csv` — baseline acoustic features.
- `processed/ravdess_metadata.csv` — dataset metadata.
- `processed/ravdess_trimming_analysis.csv` — preprocessing analysis.
- `processed/ravdess_v2_delta_rmsdiff_20261104/` — additional feature representations and trimming experiments.

### Reproduction

The experiments were developed in Google Colab using Python.

To reproduce the experiments:

1. Download the project notebooks from this GitHub repository.
2. Download the dataset and processed features from the Google Drive link above.
3. Install the dependencies listed in `requirements.txt`.
4. Configure `PROJECT_DIR` to point to the local project directory containing `data/`.
5. Execute the notebooks in numerical order.

**Note:** The current notebooks use Google Drive paths specific to the original development environment. These paths must be adapted before running the notebooks in a different environment.
