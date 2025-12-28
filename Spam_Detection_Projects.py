# SPAM DETECTION PROJECT (PYTHON + ML)

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# create a simple Dataset..

Data = {
    "message": [
        "Win money now",
        "Hello how are you",
        "Congratulations you won a lottery",
        "Meeting at 10 am",
        "Claim your free prize now",
        "Let's have lunch tomorrow",
        "Urgent! Call this number",
        "Are you coming today?"
    ],
    "label": [
        "spam",
        "ham",
        "spam",
        "ham",
        "spam",
        "ham",
        "spam",
        "ham"
    ]
}
df = pd.DataFrame(Data)

#Convert Text to Number..

Vectorizer = CountVectorizer() 
x = Vectorizer.fit_transform(df["message"])
y = df["label"]

#Split Data 

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=42)


#create Model 

model = MultinomialNB()

# Train Model
model.fit(x_train,y_train)

# test model

y_pred = model.predict(x_test)

#Accurucy

# accurucy = accuracy_score(x_train,y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accurcy" ,accuracy)

#user input prediction 

msg = input("Enter a message: ")
msg_vector = Vectorizer.transform([msg])
result = model.predict(msg_vector)

print("prediction:", result[0])
