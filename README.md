# Sentiment Analysis on Yelp Reviews
The Yelp reviews dataset consist of reviews from Yelp
The Yelp reviews full star dataset is constructed by taking randomly 130,000 training samples and 10,000 testing samples for each review star from 1 to 5. There are 2 columns in them, corresponding to class index (1 to 5) and review text.
In this project NLP sentiment analysis on yelp reviews is done using various models such as Support Vector Machine (SVM), Random Forest Classifier, Logistic Regression, XGBoost model and Naive Bayes
EDA Augmentation is also applied for the same dataset and the performance metrics for before and after augmentation is compared

### 1. Importing the relevent libraries
sklearn, numpy, matplotlib, seaborn, ntlk, pandas, string 
### 2. Loading the train and test data
### 3. Exploring the dataset
### 4. Preprocessing the data
Preprocessing involves the following steps
(i) Removing punctuations,numbers and special characters.
(ii) Converting all the words to lower alphabets
(iii) Removing all the words whose length is less than 2
(iv) Text Normalization- The process in which the sentences are broken down to words for further processing

### 5. Lemmatization
Lemmatization is the process of using vocabulary and morphological analysis of words, normally aiming to remove unnecessary endings only and to return the base or dictionary form of a word, which is known as the lemma .

### 6. Creating Word Clouds
### 7. Feature Extraction
For working of any machine learning model all the inputs has to be in feature form for the model to work. Feature Exracting involves claculating the features from the string of values using various vectorizers such as TF-IDF vectorizer or Count Vectorizer

### 8. Creating different models for calculating the accuracy
###### (1) Random Forest Classifier
A random forest is an ensemble classifier that estimates based on the combination of different decision trees. Effectively, it fits a number of decision tree classifiers on various subsamples of the dataset. Also, each tree in the forest built on a random best subset of features.
###### (2) Logistic Regression
Logistic Regression measures the relationship between a output variable Y (categorical) and one or more independent variables, which are usually continuous (but not necessarily), by using probability scores as the predicted values of the dependent variables
###### (3) XGBoost
XGBoost is the name of a machine learning method. It can help you to predict any kind of data if you have already predicted data before. We can classify any kind of data. It can be used for text classification too.
###### (4) Naive Bayes
Naive Bayes classifier (nBc) makes bold assumptions: (1) The probability of occurence independent of the probability of occurence of another word. (2) The probability of occurence of a word in a document, is independent of the location of that word within the document.
###### (5) Support Vector Machine(SVC)
In SVM the given labeled training data(supervised learning), the algorithm outputs an optimal hyperplane which categorizes the new examples.


## EDA Augmentation
Data Augmentation is the practice of synthesizing new data from data at hand.Usually, the augmented data is similar to the data that is already available.
Introduce 2 new augmented sentences for each sentence in the training set with the label 0. In each of these augmented sentences replace a maximum of 3 words by their synonyms.
