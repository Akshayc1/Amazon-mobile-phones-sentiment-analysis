from flask import Flask,flash,redirect,url_for
from flask import Flask, render_template, url_for, request, session, jsonify
from flask_wtf import FlaskForm
from werkzeug.utils import secure_filename
from wtforms import StringField, PasswordField, BooleanField, SubmitField, TextAreaField
from wtforms.validators import DataRequired
import os
from sklearn.externals import joblib
import pickle
import pandas as pd
import numpy as np

import nltk
import string
from nltk.corpus import stopwords
import re
from nltk.stem import SnowballStemmer



app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.secret_key = os.urandom(24)


model = joblib.load('../Model.pkl')

std = open('../standardization.pkl', 'rb')
stnd = pickle.load(std) 
std.close()


tf = open('../TF_IDF_model.pkl', 'rb')
vec = pickle.load(tf) 
tf.close()




class LoginForm(FlaskForm):
    username = StringField('Username:', validators=[DataRequired()])
    password = PasswordField('Password:', validators=[DataRequired()])
    submit = SubmitField('Log In')

class MLdata(FlaskForm):
	data1 = TextAreaField(u'Enter Text Here:', validators=[DataRequired()])
	submit = SubmitField('Analyze Text')


@app.route('/prediction',methods=['GET', 'POST'])
def prediction():
	form = MLdata()
	if request.method == 'POST':
		if form.validate_on_submit():
			text = str(form.data1.data)
			print(text)
			#print(type(text))
			text = text.translate(string.punctuation)
			text = text.lower().split()
			stops = set(stopwords.words("english"))
			text = [w for w in text if not w in stops and len(w) >= 3]
			text = " ".join(text)
			text = re.sub(r'http\S+', '', text)  # Removing the URL's from the text
			text = re.sub(r'www\S+', '', text)
			text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
			text = re.sub(r"what's", "what is ", text)
			text = re.sub(r"\'s", " ", text)
			text = re.sub(r"\'ve", " have ", text)
			text = re.sub(r"n't", " not ", text)
			text = re.sub(r"i'm", "i am ", text)
			text = re.sub(r"\'re", " are ", text)
			text = re.sub(r"\'d", " would ", text)
			text = re.sub(r"\'ll", " will ", text)
			text = re.sub(r",", " ", text)
			text = re.sub(r"\.", " ", text)
			text = re.sub(r"!", " ", text)
			text = re.sub(r"\/", " ", text)
			text = re.sub(r"\^", " ", text)
			text = re.sub(r"\+", " ", text)
			text = re.sub(r"\-", " ", text)
			text = re.sub(r"\=", " ", text)
			text = re.sub(r"\{", " ", text)
			text = re.sub(r"\}", " ", text)
			text = re.sub(r"'", " ", text)
			text = re.sub(r":", " : ", text)
			text = re.sub(r"e - mail", "email", text)
			text = text.split()
			stemmer = SnowballStemmer('english')
			stemmed_words = [stemmer.stem(word) for word in text]
			clean_text = " ".join(stemmed_words)
			#print(clean_text)


			#print("cjcakckac")  

			vector = vec.transform([clean_text])
			vector = stnd.transform(vector.toarray())
			pred = model.predict(vector)

			if pred[0] == 1:
				msg = "The Review is Positive. "
			else:
				msg = "The Review is Negative. "

			print(msg)
			data = [{'Prediction': msg}]
			#data = []
			#data.append(msg)
			print(data)
			
	#return redirect(url_for('main'))
	return render_template('Result.html', title='Result Page', form = form, data = data)


@app.route('/main',methods=['GET', 'POST'])
def main():
	#print("**********************************")
	form = MLdata()


	return render_template('main_page.html', title='Main Page', form = form)


@app.route('/',methods=['GET', 'POST'])
def home():
	form = LoginForm()
	if request.method == 'POST':
		if form.validate_on_submit():  # POST request with valid input?
			# Verify username and passwd
			if (form.username.data == 'sakshi' and form.password.data == '1234'):
				return redirect(url_for('main'))
			else:
				# Using Flask's flash to output an error message
				flash('Invalid username or password')

	return render_template('login.html', title='Sign In', form=form)

if __name__ == '__main__':  # Script executed directly
    app.run(debug=True)  # Launch built-in web server and run this Flask webapp
