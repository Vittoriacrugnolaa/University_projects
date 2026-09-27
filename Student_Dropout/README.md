# Student Persistence and Dropout

This notebook studies student dropout as a binary classification problem using demographic, socioeconomic, enrollment, and academic information. `Dropout` is the positive class; `Graduate` and `Enrolled` are grouped as non-dropout.

The analysis includes feature engineering, preprocessing, class-imbalance strategies, model comparison, nested cross-validation, final test evaluation, and interpretation of logistic-regression coefficients. 

[Open the notebook in Google Colab](https://colab.research.google.com/github/Vittoriacrugnolaa/University_projects/blob/main/Student_Dropout/Student_Persistence_Dropout.ipynb)

## Result

The selected pipeline combines random over-sampling with L1-regularized logistic regression.

- Nested cross-validation F1: **0.7869 +/- 0.0129**
- Training F1: **0.8003**
- Held-out test F1: **0.8142**

The small negative generalization gap indicates no evident overfitting in this run. These figures describe the binary dropout-versus-non-dropout formulation used in the notebook.

## Run locally

```bash
git clone https://github.com/Vittoriacrugnolaa/University_projects.git
cd University_projects/Student_Dropout
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Open `Student_Persistence_Dropout.ipynb` and run the cells from top to bottom. The dataset is included in `data/`; when it is not available locally, the notebook falls back to the official UCI URL.

## Files

- [Student_Persistence_Dropout.ipynb](Student_Persistence_Dropout.ipynb) - analysis and saved outputs
- [Student_Dropout_essay.pdf](Student_Dropout_essay.pdf) - short written report
- [data/student_data.csv](data/student_data.csv) - dataset used by the notebook
- [data/README.md](data/README.md) - source, license, citation, and checksum

## Data source

Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success*. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89

The dataset is distributed under the [Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/).
