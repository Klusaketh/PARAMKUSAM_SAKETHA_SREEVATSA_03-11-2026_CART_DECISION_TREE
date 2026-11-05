# CART Decision Tree — Laptop Purchase Prediction

A Jupyter notebook that uses a **CART (Classification and Regression Tree)** classifier to predict whether a customer will buy a laptop. The prediction is based on the customer's budget and the laptop's specifications.

The notebook walks through the full workflow. It builds the data, works out the best Gini split by hand, trains a scikit-learn decision tree, evaluates it, draws the tree and predicts new cases.

---

## Files

| File | Description |
|---|---|
| `cart_laptop_purchase.ipynb` | Main notebook, already executed with outputs |
| `README.md` | This file |

---

## Dataset

The notebook defines a small synthetic dataset of **16 customers** inline, so no external files are needed.

| Column | Type | Values |
|---|---|---|
| `Budget` | Feature | Low, Medium, High |
| `Processor` | Feature | Basic, Mid, Pro |
| `RAM` | Feature | 8GB, 16GB |
| `Battery` | Feature | Short, Long |
| `Purchase` | **Target** | Yes, No |

**Class balance:** 9 Yes and 7 No.

---

## Notebook Structure

1. **Load the libraries**: pandas, matplotlib and scikit-learn.
2. **Build the laptop purchase data**: creates `laptop_df` and checks shape, nulls and class counts.
3. **Check the class balance**: bar chart of Yes vs No.
4. **Work out the first CART split manually**: calculates Gini impurity for every possible `feature == value` rule and ranks them.
5. **Split and encode the data**: does a stratified train/test split, then fits a one-hot encoder on the training data only to avoid leakage.
6. **Fit the CART classifier**: `DecisionTreeClassifier(criterion="gini", max_depth=3)`.
7. **Measure performance**: accuracy, classification report and confusion matrix.
8. **Plot the decision tree**: draws the learned tree with `plot_tree`.
9. **Predict for new customers**: runs three new records through the same fitted encoder.
10. **Key points**: summary of what was learned.

---

## Gini Impurity

CART chooses splits that make the child nodes as pure as possible:

$$Gini = 1 - \sum_{k} p_k^2$$

where `p_k` is the proportion of class *k* in a node. The split with the **lowest weighted Gini**, which is the same as the **highest impurity drop**, is selected.

---

## Results

| Metric | Value |
|---|---|
| Root node Gini | 0.4922 |
| Best manual root split | `Processor == Basic` |
| Weighted Gini after split | 0.2045 |
| Impurity drop | 0.2876 |
| Train / Test rows | 12 / 4 |
| Test accuracy | 75.00% |

**Classification report (test set):**

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| No | 1.00 | 0.50 | 0.67 |
| Yes | 0.67 | 1.00 | 0.80 |

**Learned tree (summary):**

```
Processor = Basic?
├── Yes → No purchase
└── No  → Budget = Low?
          ├── No  → Purchase
          └── Yes → Processor = Mid?
                    ├── Yes → Purchase
                    └── No  → No purchase
```

**Sample predictions:**

| Budget | Processor | RAM | Battery | Predicted | Probability |
|---|---|---|---|---|---|
| High | Pro | 16GB | Long | Yes | 100.0% |
| Low | Basic | 8GB | Short | No | 100.0% |
| Medium | Mid | 16GB | Short | Yes | 100.0% |

---

## Requirements

- Python 3.9+
- pandas
- matplotlib
- scikit-learn **1.2 or newer**, because the notebook uses the `sparse_output` argument of `OneHotEncoder`
- Jupyter Notebook or JupyterLab, because the notebook calls `display()`

Install:

```bash
pip install pandas matplotlib scikit-learn notebook
```

---

## How to Run

```bash
jupyter notebook cart_laptop_purchase.ipynb
```

Then choose **Kernel → Restart & Run All**. The notebook also runs as-is in Google Colab.

---

## Key Variables

| Variable | Purpose |
|---|---|
| `laptop_df` | Full dataset as a DataFrame |
| `label_col`, `predictor_cols` | Target column name and feature column names |
| `gini_impurity()` | Calculates the Gini impurity of a subset |
| `splits_table` | All candidate root splits ranked by weighted Gini |
| `encoder_pipeline` | `ColumnTransformer` with `OneHotEncoder`, fitted on training data only |
| `laptop_tree` | Trained `DecisionTreeClassifier` |
| `upcoming_buyers` | New customer records used for prediction |

---

## Limitations

- The dataset is only 16 rows, so with 4 test rows the scores can change a lot between different random splits.
- The data is synthetic and meant only for learning how CART works.
- A real purchase prediction model would need much more data, cross-validation and hyperparameter tuning.
