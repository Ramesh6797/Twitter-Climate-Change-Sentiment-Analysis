# Twitter Sentiment Analysis on Climate Change

This project analyzes sentiments from tweets concerning climate change. Using Natural Language Processing (NLP) techniques and Machine Learning, this model classifies tweets into distinct sentiment categories (e.g., Pro-climate, Anti-climate, Neutral).

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [How to Run This Project](#how-to-run-this-project)

---

## Project Overview

The goal of this project is to build and evaluate a machine learning model capable of performing sentiment analysis on Twitter data related to climate change. This analysis can help in understanding public opinion and discourse on this critical issue. The notebook covers all the essential steps from data loading and cleaning to model training and evaluation.

---

## Dataset

The dataset used in this project is a collection of tweets related to climate change, sourced from Twitter. Each tweet in the dataset is labeled with a sentiment.

* **Source:** ( Kaggle )
* **Size:** ( 43943 tweets )
* **Labels:** The tweets are classified into the following categories:
    * `1` (Pro-climate change / Believer)
    * `0` (Neutral)
    * `-1` (Anti-climate change / Denier)
    * `2` (News)

---

## Methodology

The project follows a standard machine learning pipeline for NLP tasks:

1.  **Data Loading:** The dataset is loaded into a pandas DataFrame.
2.  **Data Cleaning & Preprocessing:**
    * Removing URLs, mentions, and hashtags.
    * Converting text to lowercase.
    * Tokenization (splitting text into individual words).
    * Removing stopwords (common words like 'the', 'a', 'is').
    * Lemmatization/Stemming to reduce words to their root form.
3.  **Feature Extraction:**
    * The cleaned text data is converted into numerical vectors using the **TF-IDF (Term Frequency-Inverse Document Frequency)** technique. This allows the machine learning model to process the text.
4.  **Model Training:**
    * The dataset is split into training and testing sets.
    * Seven different classification models were trained and evaluated to find the best performer for this sentiment analysis task
5.  **Model Evaluation:**
    * The model's performance is evaluated on the test set using metrics like **Accuracy, Recall, Precision and the F1-Score**. A confusion matrix is also generated to visualize the results.

---

## Results and Model Comparison

Seven different classification models were trained and evaluated to find the best performer for this sentiment analysis task. The performance metrics for each model on the test set are detailed in the table below.

The **Logistic Regression** model provided the highest overall performance, achieving the best Accuracy and F1-Score. It was therefore selected as the most effective model for this particular problem.

### Model Performance Table

| Model                                  | Accuracy | Recall   | Precision| F1-Score |
| -------------------------------------- | :------: | :------: | :------: | :------: |
| **Logistic Regression** | **0.7338** | **0.7338** | **0.7273** | **0.7258** |
| Linear Support Vector Classification   |  0.7301  |  0.7301  |  0.7331  |  0.7143  |
| Ridge Classification                   |  0.7194  |  0.7194  |  0.7126  |  0.7127  |
| Extra Trees Classification             |  0.7079  |  0.7079  |  0.7090  |  0.6907  |
| Random Forest Classification           |  0.7044  |  0.7044  |  0.7120  |  0.6850  |
| Decision Tree                          |  0.6070  |  0.6070  |  0.6193  |  0.5582  |
| K Neighbors Classification             |  0.3928  |  0.3928  |  0.6953  |  0.4042  |

*(Note: The scores have been rounded to four decimal places for better readability.)*
---

## Technologies Used

* **Language:** Python 3
* **Libraries:**
    * `pandas` for data manipulation and analysis.
    * `NumPy` for numerical operations.
    * `NLTK` & `spaCy` for natural language processing tasks.
    * `scikit-learn` for machine learning models and metrics.
    * `Matplotlib` & `Seaborn` for data visualization.
    * `Google Colab` as the development environment.

---

## How to Run This Project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/Twitter-Climate-Change-Sentiment-Analysis.git](https://github.com/your-username/Twitter-Climate-Change-Sentiment-Analysis.git)
    ```
2.  **Open the Notebook:** Upload and open the `Twitter_Climate_Change_Sentiment_Analysis.ipynb` notebook in Google Colab or a local Jupyter Notebook environment.
3.  **Run the Cells:** Execute the cells sequentially to see the entire workflow from data loading to model evaluation.
