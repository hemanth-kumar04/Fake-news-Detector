# Fake News Detection Using NLP and Machine Learning

This project focuses on detecting fake news using Natural Language Processing (NLP) and Machine Learning (ML) techniques. The goal is to build a machine learning model that can classify news articles as either **Fake** or **Real** based on the content of the article.

## Project Overview

Fake news is a growing concern, especially on social media and news websites. This project aims to tackle the problem of fake news detection by using text data from news articles and applying machine learning algorithms to predict whether a news article is fake or real.

### Key Features:
- **Data Collection:** We use two datasets—`fake.csv` and `true.csv`—containing fake and real news articles respectively.
- **Text Preprocessing:** The text data is cleaned by removing non-alphabetic characters, tokenizing the text, and removing stopwords to prepare it for further analysis.
- **Feature Extraction:** We use TF-IDF (Term Frequency-Inverse Document Frequency) to convert the text data into numerical features for machine learning.
- **Modeling:** Logistic Regression is used to train the model for classification.
- **Evaluation:** The model's performance is evaluated using accuracy and classification reports.
- **Prediction:** The trained model can predict whether a news article is fake or real based on new input text.

## Technologies Used

- **Python**: The main programming language used for this project.
- **Libraries**:
  - `pandas`: For data manipulation and loading CSV files.
  - `nltk`: For text preprocessing (tokenization, stopwords removal).
  - `scikit-learn`: For machine learning algorithms, feature extraction, and model evaluation.
  - `joblib`: To save and load the trained machine learning model and vectorizer.
- **Text Mining**: TF-IDF vectorizer to convert text into numerical features.
- **Logistic Regression**: The machine learning algorithm used for classification.

## Installation

To run this project locally, you'll need to install the required Python libraries. You can do this by running:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should include the following libraries:

```
pandas
nltk
scikit-learn
joblib
```

Alternatively, you can install the individual libraries using `pip`:

```bash
pip install pandas nltk scikit-learn joblib
```

## How to Use

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/fake-news-detection.git
   ```

2. Navigate to the project folder:

   ```bash
   cd fake-news-detection
   ```

3. Download the necessary NLTK datasets (if you haven't already):

   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```

4. Run the script to train the model:

   ```bash
   python fnd.py
   ```

5. After training, the model and vectorizer will be saved as `fake_news_detector_model.pkl` and `tfidf_vectorizer.pkl` respectively.

6. To make predictions on new data, modify the script or use the trained model and vectorizer to transform and classify new text.

## Sample Usage for Prediction:

After training the model, you can use it to predict whether a new news article is fake or real:

```python
# Load the saved model and vectorizer
import joblib

model = joblib.load('fake_news_detector_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Sample text for prediction
text = ["Donald Trump just couldn't wish all Americans a Happy New Year and leave it at that."]

# Transform the text using the saved vectorizer
text_vectorized = vectorizer.transform(text)

# Make a prediction
prediction = model.predict(text_vectorized)

# Print the result
if prediction == 0:
    print("This is Fake News!")
else:
    print("This is Real News!")
```

## Dataset Information

- **Fake News Dataset**: Contains news articles labeled as fake (1).
- **True News Dataset**: Contains news articles labeled as real (0).

Both datasets consist of the following columns:
- **title**: The title of the news article.
- **text**: The main content of the article.
- **subject**: The subject/category of the article.
- **date**: The publication date of the article.
- **label**: 0 for real news, 1 for fake news.

## Model Evaluation

- The model is evaluated using **accuracy score** and **classification report**, which includes precision, recall, and F1 score.

## Conclusion

The model can be used to classify news articles as fake or real. This is a step forward in tackling the spread of misinformation, but further improvements can be made by exploring more advanced NLP techniques and machine learning models.

## License

This project is licensed under the MIT License.

## Acknowledgments

- Thanks to the creators of the datasets used for this project.
- NLTK for providing the necessary NLP tools.

