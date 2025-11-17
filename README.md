Amazon Alexa Review: Sentiment Analysis & NLP Tasks
This project performs a comprehensive analysis of Amazon Alexa reviews, moving from basic sentiment classification to advanced NLP tasks like star rating regression and Retrieval-Augmented Generation (RAG) for question answering.

Project Overview
The notebook explores various techniques to understand and model customer feedback from the amazon_alexa.tsv dataset. The project is structured into three main modeling tasks:

Sentiment Classification: Classifying reviews as positive (1) or negative (0).

Rating Regression: Predicting the 1-5 star rating based on the review text.

RAG Q&A: Building a system to answer questions about the reviews using a local LLM (Mistral-7B) and a vector database.

📊 Dataset
File: amazon_alexa.tsv

Target Variables:

feedback: Binary sentiment (1 for positive, 0 for negative).

rating: Numerical star rating (1 to 5).

Key Feature: verified_reviews (the raw text of the customer review).

🚀 Project Pipeline
1. Exploratory Data Analysis (EDA)

Nulls & Duplicates: The dataset was cleaned by removing null values and duplicates.

Class Imbalance: A key finding was the severe class imbalance in the feedback column, with ~91% of reviews being positive. This imbalance was addressed in the modeling phase using class_weight='balanced'.

Distributions: Visualized the distribution of star ratings (heavily skewed to 5 stars) and product variation counts.

2. Text Preprocessing

A standard NLP preprocessing pipeline was applied to the raw verified_reviews to create the final_cleaned_text column:

Remove Special Characters: All non-alphanumeric characters were removed using regex.

Lowercase: Text was converted to lowercase.

Remove Stopwords: Common English stopwords (e.g., "the", "is", "a") were removed using NLTK.

Lemmatization: Words were reduced to their root form using NLTK's WordNetLemmatizer (e.g., "loved" -> "love").

3. Feature Engineering (Text Vectorization)

Several vectorization methods were explored to convert text into numerical features:

Traditional Methods:

Bag-of-Words (BoW): CountVectorizer (Top 1000 features).

TF-IDF: TfidfVectorizer (Top 1000 features).

N-grams: CountVectorizer with ngram_range=(1, 2) (Top 1000 features).

Word Embeddings (Average Vectorization):

Word2Vec: Trained custom CBOW and Skip-gram models on the review corpus.

GloVe: Used pre-trained glove.6B.100d.txt vectors.

Sentence-Transformer Embeddings:

all-MiniLM-L6-v2: Used for classification and RAG.

all-mpnet-base-v2: Used for the regression task.

🤖 Modeling & Results
Task 1: Sentiment Classification (Positive vs. Negative)

Multiple models were trained to predict the binary feedback label.

Models: RandomForestClassifier, MultinomialNB, LogisticRegression.

Best Traditional Model: MultinomialNB using N-gram features achieved the highest Macro F1-score (0.81).

Best Overall Model: A LogisticRegression model trained on all-MiniLM-L6-v2 Sentence-Transformer embeddings with class_weight='balanced' performed best at handling the minority class (negative reviews).

Accuracy: 82.38%

Class 0 (Negative) Recall: 0.83 (Successfully identified 83% of all negative reviews).

Task 2: Star Rating Prediction (Regression)

A Support Vector Regressor (SVR) was trained to predict the 1-5 rating from the review text.

Model: SVR with an RBF kernel.

Features: all-mpnet-base-v2 embeddings.

Result: The model achieved a Mean Squared Error (MSE) of 0.496, demonstrating a strong ability to predict the star rating from text.

Task 3: RAG Question Answering System

A Retrieval-Augmented Generation (RAG) pipeline was built to answer questions about the reviews.

Embedding Model: all-MiniLM-L6-v2 (for creating text embeddings).

Vector Store: ChromaDB (for storing and retrieving relevant review chunks).

LLM: Mistral-7B-Instruct-v0.2-GGUF (run locally using LlamaCpp).

This system can successfully answer queries by finding relevant reviews and synthesizing an answer, as shown in the test questions (e.g., "Do users like the sound quality?").

🛠️ Technologies Used
Data Analysis: pandas, numpy

Visualization: matplotlib, seaborn

NLP & Preprocessing: nltk, re

ML & Vectorization: scikit-learn (CountVectorizer, TfidfVectorizer, LogisticRegression, SVR, RandomForest, MultinomialNB)

Embeddings: gensim (Word2Vec), sentence-transformers

RAG Pipeline: langchain, llama-cpp-python, chromadb
