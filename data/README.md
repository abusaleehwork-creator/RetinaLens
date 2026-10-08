# Dataset documentation

This project uses the UCI Machine Learning Repository dataset: "Diabetic Retinopathy Debrecen Data Set" (dataset ID 329).

## Source
- UCI Machine Learning Repository
- Dataset page: https://archive.ics.uci.edu/dataset/329/diabetic+retinopathy+debrecen+data+set
- Data source: retinal-image-derived features extracted from fundus images, with a binary label indicating whether diabetic retinopathy signs are present.

## Dataset metadata
- Dataset name: Diabetic Retinopathy Debrecen Data Set
- Dataset ID: 329
- Observations: 1151
- Features: 19 input features
- Target column: `Class`
- Target definition: `1 = contains signs of diabetic retinopathy`, `0 = no signs of diabetic retinopathy`

## Feature overview
The dataset consists of image-derived measurements, including:
- quality assessment and pre-screening indicators
- microaneurysm count features (`ma1` to `ma6`)
- exudate features (`exudate1` to `exudate8`)
- macula-to-optic-disc distance
- optic-disc diameter
- AM/FM classification flag

## Important limitations
- This is a tabular dataset derived from retinal image analysis, not raw images.
- The source page lists the dataset as a research benchmark; no explicit open-source license was found on the UCI page, so the project uses it strictly for educational and research learning.
- Some features are close to the final label or strongly related to preprocessing decisions (for example, `pre_screening` and `am_fm_classification`). These are discussed in the model training notes and excluded in the leakage-aware feature configuration.
- This project is educational and does not constitute medical advice or clinical validation.

## Storage in this project
The raw dataset is fetched programmatically from the UCI repository and stored in `data/retina_data.csv` when the project is set up.
