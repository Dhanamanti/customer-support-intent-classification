# Customer Support Intent Classification Using Machine Learning

## Project Overview

Customer support teams receive a large number of messages every day. Manually identifying the purpose of each message can be time-consuming. This project uses Natural Language Processing (NLP) and Machine Learning to automatically classify customer messages into the appropriate support intent.

The model takes a customer message as input, converts the text into numerical features using **TF-IDF**, and predicts the most likely intent using **Logistic Regression**. GridSearchCV is used to tune the model's hyperparameters and evaluate whether its performance improves.

The project covers the complete workflow, from dataset preparation and text vectorization to model training, evaluation, hyperparameter tuning, and live prediction.

## Objectives

- Prepare and validate a customer support intent dataset.
- Convert text messages into numerical features using TF-IDF.
- Train a Logistic Regression classification model.
- Evaluate performance using accuracy, precision, recall, and F1-score.
- Apply GridSearchCV to identify a suitable hyperparameter configuration.
- Predict the intent of new customer messages.

## Dataset

**Source:** [Customer Support Intent Classification Dataset — Kaggle](https://www.kaggle.com/datasets/scodepy/customer-support-intent-dataset)

The dataset contains customer utterances and their corresponding intent labels, categories, and tags.

| Dataset detail | Value |
|---|---|
| Total records after combining CSV files | 8,175 |
| Input feature | Customer utterance |
| Target variable | Intent |
| Number of unique intents | 27 |
| Missing values found | 0 |
| Exact duplicate rows found | 0 |
| Train-test split | 80:20 |
| Training records | 6,540 |
| Testing records | 1,635 |

The dataset was checked for missing values and duplicate records before model development.

## Technologies Used

- **Python** — model development and data processing
- **Pandas** — dataset loading and manipulation
- **Scikit-learn** — machine learning, model evaluation, and hyperparameter tuning
- **TF-IDF Vectorizer** — text feature extraction
- **Logistic Regression** — intent classification
- **GridSearchCV** — hyperparameter optimization
- **Google Colab** — development and experimentation
- **GitHub** — project versioning and sharing

## Project Workflow

### 1. Data Preparation
Loaded and combined the three CSV files into a single dataset. Checked the dataset for missing values, duplicate records, and missing intent labels.

### 2. Train-Test Split
Split the dataset into 80% training data and 20% testing data. Stratified splitting was used to preserve the approximate distribution of intent classes.

### 3. Text Vectorization with TF-IDF
Used `TfidfVectorizer` to convert customer messages into numerical features.

Configuration:
- Lowercase conversion enabled
- Unigrams and bigrams: `ngram_range=(1, 2)`
- English stop-word removal enabled

The vectorizer was fitted on the training data and then used to transform the test data.

### 4. Baseline Model Training
Trained a Logistic Regression classifier using the TF-IDF features. The baseline model was configured with `max_iter=1000`.

### 5. Model Evaluation
Evaluated predictions on the held-out test dataset using accuracy, precision, recall, and F1-score.

### 6. Hyperparameter Tuning
Used `GridSearchCV` with five-fold cross-validation to compare the following values of the Logistic Regression parameter `C`:

`C = [0.1, 1, 10]`

The best parameter identified was `C = 10`.

### 7. Live Prediction
Tested the trained model on a new customer message to demonstrate intent prediction.

## Model Performance

The baseline and tuned models were evaluated on the same held-out test set.

| Metric | Baseline Model | Tuned Model |
|---|---:|---:|
| Test accuracy | 97.49% | 97.92% |
| Best `C` parameter | Default | 10 |

The tuned model improved test accuracy by **0.43 percentage points** compared with the baseline model.

**Best cross-validation accuracy:** 97.89%

The cross-validation score and test accuracy are different measurements: cross-validation is calculated across folds of the training data, while test accuracy is calculated on the held-out test set.

### Evaluation Metrics

- **Accuracy:** The proportion of all test messages classified correctly.
- **Precision:** The proportion of predictions for an intent that were correct.
- **Recall:** The proportion of actual messages belonging to an intent that were identified correctly.
- **F1-score:** The harmonic mean of precision and recall.

Although the overall accuracy is high, performance varies across intents. Invoice-related intents, particularly `check_invoice` and `get_invoice`, remain challenging and could benefit from further investigation.

## Sample Prediction

**Input message:**

`I need help cancelling my recent order`

**Predicted intent:**

`cancel_order`

This example demonstrates how the model classifies a new customer message into one of the supported intent labels.

## How to Run the Project

### Option 1: Google Colab

1. Download or open the Jupyter Notebook included in this repository.
2. Upload the notebook to [Google Colab](https://colab.research.google.com/).
3. Obtain the dataset from the Kaggle source listed above.
4. Upload or extract the dataset in Colab and update the file paths if necessary.
5. Run the notebook cells in order, from data preparation through model evaluation and prediction.

### Option 2: Local Environment

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/customer-support-intent-classification.git
```

Move into the project directory:

```bash
cd customer-support-intent-classification
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Open the notebook using Jupyter Notebook or JupyterLab and run the cells in order. Make sure the dataset is downloaded and placed in the location expected by the notebook.

Replace `YOUR-USERNAME` with your GitHub username before using the clone command.

## Repository Structure

```text
customer-support-intent-classification/
├── README.md
├── requirements.txt
└── customer_support_intent_classification.ipynb
```

The notebook contains the data preparation, TF-IDF transformation, model training, baseline evaluation, hyperparameter tuning, tuned-model evaluation, and sample prediction.

## Limitations and Future Improvements

- Investigate confusion between similar invoice-related intents.
- Compare Logistic Regression with other text classification algorithms.
- Explore additional hyperparameters and cross-validation strategies.
- Analyze a confusion matrix to identify frequently confused intent pairs.
- Test the model on new, real-world customer messages.
- Develop a simple web interface for interactive intent prediction.

## Conclusion

This project demonstrates an end-to-end text classification workflow using NLP and Machine Learning. The tuned Logistic Regression model achieved **97.92% test accuracy** on the held-out dataset, improving on the baseline result of 97.49%.

The project also highlights the importance of evaluating individual intent classes rather than relying on overall accuracy alone.

## Acknowledgements

- Dataset: [Customer Support Intent Classification — Kaggle](https://www.kaggle.com/datasets/scodepy/customer-support-intent-dataset)
- Libraries: Pandas and Scikit-learn
- Development environment: Google Colab
