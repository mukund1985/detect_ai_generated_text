# LLM - Detect AI Generated Text

Building a model to identify which essay was written by middle and high school students, and which was written using a large language model? With the spread of LLMs, many people fear they will replace or alter work that would usually be done by humans. Educators are especially concerned about their impact on students’ skill development, though many remain optimistic that LLMs will ultimately be a useful tool to help students improve their writing skills.

At the forefront of academic concerns about LLMs is their potential to enable plagiarism. LLMs are trained on a massive dataset of text and code, which means that they are able to generate text that is very similar to human-written text.

For example, students could use LLMs to generate essays that are not their own, missing crucial learning keystones. This work can help to identify telltale LLM artifacts and advance the state of the art in LLM text detection.

By using texts of moderate length on a variety of subjects and multiple, unknown generative models, I aim to replicate typical detection scenarios and incentivize learning features that generalize across models.

## Approach

I am going to develop a model that can accurately classify essays as either student-written or LLM-generated.

Given the dataset details, here's a structured approach:

### 1. Data Preprocessing and Exploration

- **Load and Examine Data**: Import `{test|train}_essays.csv` and `train_prompts.csv`. Understand the structure, examine any missing values, and get a feel for the data.
- **Text Preprocessing**: Clean and preprocess the essay texts. This may include lowercasing, removing punctuation, lemmatization and handling Markdown format in `source_text`.

### 2. Feature Engineering

- **Textual Feature**: Extract feature like word count, sentence length, lexical diversity, syntactic complexity and use of numerals referrign to paragraphs.
- **Prompt Matching**: Analysing how closely each essay adheres to its corresponding prompt in terms of thematic relvance and use of specific terms or concepts from the source texts.
- **NLP Techniques**: Employing techniques like TF-IDF, word embeddings(Word2vec,Glove), or more advanced NLP models to transform text into a format suitable for machine learning.

### 3. Model Development

- **Baseline Model**: Starting with a smiple model, like Logistic Regression or Naive Bayes, as a baseline.
- **Advanced Models**: Experiment with more complex models like Random Forest, Gradient Boosting Machines, or deep learning models(LSTM, Transformers).
- **Model Training**: Train models on the provided training set. Remember that the training set is imblanaced with fewer LLM-generated essays.

### 4. Model Evaluation and Selection

- **Cross-Validation**: Use cross-validation to assess model performance and mitigate overfitting.
- **Metrics**: Focus on metrics like accuracy, precision, recall, F1-score, and ROC-AUC.
- **Error Analysis**: Analyse the types of errors your model makes to understand its weaknesses.

### 5. Adressing Class Imbalance

- Since the training set is imbalanced, will consider techiques like `SMOTE`, `class weight adjustment`, or `undersampling` to handle this.

### 6. Additional Data Generation

- Generating more LLM-written essays for training purposes using different LLMs. Ensuring diversiy in the generation process to cover a wide range of styles and topics.

### 7. Model Tunning and Optimization

- **Hyperparameter Tunning**: Using techniques like grid search or random search to find the optimal model parameters.
- **Feature Selection**: Identifying and retaining the most informative features.

### 8. Final Model Training and Testing

- Train your chosen model on the entire training set.
- Make predictions on `test_essay.csv`.

#### Tools and Libraries

- `Python` with libraries `Pandas`, `NumPy`, `Scikit-Learn` for data handling and modeling.
- `NLTK`, `SpaCy`, or `transformers` for NLP tasks.
- `TensorFlow` or `PyTorch` for deep learning models(if used).

## DataSet

Dataset comprises about 10,000 essays, some written by students and some generated by a variety of large language models (LLMs). The goal of this model is to determine whether or not easay was generated by an LLM.

All of the essays were written in response to one of seven essay prompts. In each prompt, the student were instructed to read one or more source texts and then write a response. This same information may or may not have been provided as input to an LLM when generating an essay.

Essay from two of the prompts compose the training set; the remaining essays compose the hidden test set. Nearly all of the training set essays were written by students, with only a few generated essays given as examples. We may wish to generate more essays to use as training data.

The data in test_essays.csv is only dummy data to help to author solution. When submission is scored, this example test data will be replaced with the full test set. There are about 9000 essays in the test set, both student written and LLM generated.

- `train_essay.csv` and `test_essays.csv` to understand the essays' dataset structure.
- `train_prompts.csv` to see the prompts given for the essays.
- `sample_submission.csv` to understand the format required for submission.

Here's an overview of the contents of the provided files:

- **Train Essays** (`train_essays.csv`):
  - Contains columns for id (unique identifier for each essay), prompt_id (which prompt the essay responds to), text (the essay content), and generated (indicates whether the essay is student-written or LLM-generated, with 0 for student and 1 for LLM).
  - Example:
    - `id`: 0059830c
    - `prompt_id`: 0
    - `text`: [Essay Content]
    - `generated`: 0
- **Test Essays** (`test_essays.csv`):
  - Similar structure to train essays, but without the `generated` column. It includes `id`, `prompt_id`, and `text`.
  - Example:
    - `id`: 0000aaaa
    - `prompt_id`: 2
    - `text`: Aaa bbb ccc.
- **Train Prompts** (`train_prompts.csv`):
  - Contains `prompt_id`, `prompt_name`, `instructions`, and `source_text` (in Markdown format).
  - Example:
    - `prompt_id`: 0
    - `prompt_name`: Car-free cities
    - `instructions`: [Instructions for the essay]
    - `source_text`: [Source text in Markdown]
- **Sample Submission** (`sample_submission.csv`):
  - Format for submissions, with `id` and a `generated` score (likely a probability of being LLM-generated).
  - Example:
    - `id`: 0000aaaa
    - `generated`: 0.1

## Folder Structure

This project follows the following folder structure:

```
project_name/
│
├── data/ # Data files
│ ├── raw/ # Original, immutable data dump
│ ├── processed/ # Processed data ready for analysis
│ └── external/ # Data from third-party sources
│
├── notebooks/ # Jupyter notebooks for exploration and presentation
│
├── src/ # Source code for this project
│ ├── init.py # Makes src a Python module
│ ├── data/ # Scripts to download or generate data
│ ├── features/ # Scripts to turn raw data into features for modeling
│ ├── models/ # Scripts to train and use models for predictions
│ └── visualization/ # Scripts for exploratory and results-oriented visualizations
│
├── output/ # Output files like figures, logs, etc.
│
├── tests/ # Unit tests
│
├── requirements.txt # The requirements file for reproducing the analysis environment
├── .gitignore # Lists files and folders to be ignored by Git
└── README.md # Top-level README for developers using this project

```
