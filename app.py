import os
import matplotlib.pyplot as plt
import cv2
from flask import Flask
import easyocr
from pylab import rcParams
from IPython.display import Image
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from flask import Flask, request, render_template
from flask_ngrok import run_with_ngrok
import pickle
import numpy as np


from flask import send_from_directory
from flask import Flask
from flask import send_from_directory
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from os import XATTR_CREATE


rcParams['figure.figsize']=8,16
reader=easyocr.Reader(['en'])

loaded_model_1 = pickle.load(open('march_w2v4epo.sav', 'rb'))
loaded_model_2 = pickle.load(open('march_token1epo.sav', 'rb'))
loaded_model_3 = pickle.load(open('march_seq1epo.sav', 'rb'))
UPLOAD_FOLDER = '/content/drive/MyDrive/Colab Notebooks/modelsave/static/uploads'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def ind():
	return render_template('Index.html')
 
@app.route('/upload')
def upload_form():
	return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_image():
  if 'file' not in request.files:
    flash('No file part')
    return redirect(request.url)
  file = request.files['file']
  if file.filename == '':
    flash('No image selected for uploading')
    return redirect(request.url)
  if file and allowed_file(file.filename):
    output=reader.readtext(path)
    op=""
    for i in range(0,len(output)):
      op=op+str(output[i][1])+" "
    op.strip()
    x=[op]
    maxlen=1000
    x=loaded_model_2.texts_to_sequences(x)
    x=pad_sequences(x,maxlen=maxlen)
    pred = (loaded_model_3.predict(x) >=0.5).astype(int)
    if pred==0:
      flash('Potentially Unwanted Information')
    else:
      flash('Shareable Information')
    return render_template('upload.html', filename=filename)
  else:
    flash('Allowed image types are -> png, jpg, jpeg, gif')
    return redirect(request.url)





@app.route('/tx', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
      x=[]
      y = request.form['info']
      x.append(y)
      maxlen=1000
      x=loaded_model_2.texts_to_sequences(x)
      x=pad_sequences(x,maxlen=maxlen)
      pred = (loaded_model_3.predict(x) >=0.5).astype(int)
      if pred==0:
        return render_template('tx.html', pred="Potentially Unwanted Information")
      else:
        return render_template('tx.html', pred="Shareable Information")

   

    return render_template('tx.html')

if __name__ == '__main__':
    app.run(debug=True)



