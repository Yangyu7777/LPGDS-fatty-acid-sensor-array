# L-PGDS Fatty-Acid Sensor Array

Reproducible analysis code for the L-PGDS/DPH fatty-acid sensor array.

## Workflow

Six sensor channels are analyzed: **WT, LA, LB, LC, LD, and LE**.

For each discrimination task, every analyte class contains 10 experimental replicates. The analysis uses a reproducible stratified random split:

- 6 replicates per class for training
- 4 replicates per class as held-out prediction samples

To avoid information leakage:

1. `StandardScaler` is fitted on the training set only.
2. Linear discriminant analysis (LDA) is fitted on the training set only.
3. Held-out samples are transformed with the training-fitted scaler and classified by the training-fitted LDA model.
4. Prediction accuracy is calculated only from held-out samples.
5. The 95% covariance ellipses are calculated from training LDA scores only.

Bar plots are generated from the dedicated mean ± SD worksheets.

## Files

- `analysis_pipeline.py`: complete analysis and plotting workflow
- `config.example.json`: worksheet configuration
- `requirements.txt`: Python dependencies
- `VALIDATION.md`: validation results obtained with the working dataset
- `.gitignore`: excludes local environments, generated output, and Excel data by default

## Expected Excel workbook

Default worksheet pairs:

| Analysis | Raw/LDA worksheet | Mean ± SD worksheet |
|---|---|---|
| Unsaturation | `不饱和度LDA` | `不饱和度柱状图` |
| Chain length | `不同链长LDA` | `不同链长柱状图` |
| Mixture ratio | `不同比例LDA` | `不同比例柱状图` |
| C18 concentration | `C18浓度LDA` | `C18浓度柱状图` |
| C18:2 concentration | `C18-2浓度LDA` | `C18-2浓度柱状图` |

### Raw/LDA worksheets

Seven columns without a header row:

```text
class_label | WT | LA | LB | LC | LD | LE
```

### Mean ± SD worksheets

The first row contains channel labels. Subsequent rows contain:

```text
condition |
WT_mean | WT_SD |
LA_mean | LA_SD |
LB_mean | LB_SD |
LC_mean | LC_SD |
LD_mean | LD_SD |
LE_mean | LE_SD
```

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python analysis_pipeline.py \
  --input "20260921-ALL区分汇总LDA和柱状图.xlsx" \
  --output-dir analysis_output \
  --seed 20260921
```

A custom worksheet configuration can be supplied with:

```bash
python analysis_pipeline.py \
  --input data.xlsx \
  --output-dir analysis_output \
  --config config.example.json
```

## Outputs

```text
analysis_output/
├── figures/
│   ├── unsaturation_LDA.png
│   ├── unsaturation_LDA.svg
│   ├── unsaturation_bar.png
│   ├── unsaturation_bar.svg
│   └── ...
├── all_figures.pdf
├── lda_summary.csv
├── lda_predictions_and_split.csv
├── bar_summary_consistency_check.csv
└── analysis_metadata.json
```

The LDA figures use filled circles for training samples and hollow circles for held-out prediction samples. Axis limits explicitly include the complete rotated 95% ellipse boundaries, preventing edge ellipses from being clipped.

## Reproducibility

The default base random seed is `20260921`. Each analysis receives a deterministic offset from this seed. The exact training/prediction assignment and LDA coordinates are saved in `lda_predictions_and_split.csv`.
