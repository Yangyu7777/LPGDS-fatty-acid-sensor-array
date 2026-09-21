# Validation

The analysis workflow was executed successfully against the working Excel workbook using base seed `20260921` and a 6:4 train/prediction split within each class.

## Held-out prediction summary

| Analysis | Correct / held-out | Accuracy |
|---|---:|---:|
| Unsaturation | 12 / 12 | 100% |
| Chain length | 16 / 16 | 100% |
| Mixture ratio | 20 / 20 | 100% |
| C18 concentration | 20 / 20 | 100% |
| C18:2 concentration | 20 / 20 | 100% |

## Quality-control note

The script also compares the dedicated bar-plot mean ± SD values with statistics recomputed from the corresponding raw 10-replicate LDA worksheets. In the working workbook, six channel-condition entries were flagged as non-identical at a tolerance of 1e-6. These differences are reported rather than silently overwritten.
