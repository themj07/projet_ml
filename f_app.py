
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# loading the data from csv file to a pandas Dataframe
raw_mail_data = pd.read_csv('D:\FLUTTER\ML\mail_data.csv')

# replace the null values with a null string
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')

# label spam mail as 0;  ham mail as 1;

mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1

# separating the data as texts and label

X = mail_data['Message']

Y = mail_data['Category']


# Splitting the data into training data & test data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=3)

# transform the text data to feature vectors that can be used as input to the Logistic regression
feature_extraction = TfidfVectorizer(
    min_df=1, stop_words='english', lowercase=True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


model = LogisticRegression()

# training the Logistic Regression model with the training data
model.fit(X_train_features, Y_train)

# prediction on training data

prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(
    Y_train, prediction_on_training_data)


@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    input_mail = [request.json['input_mail']]  # Mettez le texte dans une liste

    # convert text to feature vectors
    input_data_features = feature_extraction.transform(input_mail)

    # making prediction
    prediction = model.predict(input_data_features)

    if prediction[0] == 1:
        return jsonify({'result': 'Ham mail'})
    else:
        return jsonify({'result': 'Spam mail'})


if __name__ == '__main__':
    app.run(debug=True)
