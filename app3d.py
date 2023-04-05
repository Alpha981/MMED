from flask import Flask,render_template,request,redirect
import sys
import os
from os.path import join, dirname, realpath
from werkzeug.utils import secure_filename
import cv2
import tensorflow as tf
from keras.preprocessing import image
from keras.models import load_model,model_from_json
import numpy as np
import jinja2
import aspose.threed as a3d
from optimizer import Optimizer
from config import Config
config = Config()
config.fillFromDicFile('optimConfig.ini')
config.device = 'cpu'
config.path = 'baselMorphableModel'
UPLOAD_FOLDER = 'static/'
outputDir = 'static/output/' 
optimizer = Optimizer(outputDir ,config) 

app = Flask(__name__,template_folder='templates/',static_folder='static/')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
app.secret_key = 'sameer'

@app.route('/')
def main():
    return render_template('index1.html')

@app.route('/predict',  methods=['POST', 'GET'])
def uploadFile():
    if request.method == 'POST':
        if 'uploaded-file' not in request.files:
            return render_template('index.html')
        uploaded_img = request.files['uploaded-file']
        if uploaded_img.filename == '':
            return render_template('index.html')
        uploaded_img.save('static/file.jpg')
        img1 = cv2.imread('static/file.jpg')
        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = cascade.detectMultiScale(gray, 1.1, 3)
        for x,y,w,h in faces:
            cv2.rectangle(img1, (x,y), (x+w, y+h), (0,255,0), 2)
            cropped = img1[y:y+h, x:x+w]
        cv2.imwrite('static/after.jpg', img1)
        try:
            cv2.imwrite('static/cropped.jpg', cropped)
        except:
            pass
        try:
            image = cv2.imread('static/cropped.jpg', 0)
        except:
            image = cv2.imread('static/file.jpg', 0)
        image = tf.keras.utils.load_img('static/cropped.jpg',target_size = (48,48),color_mode = "grayscale")
        image = np.array(image)
        image = image/255.0
        image = np.reshape(image, (1,48,48,1))
        model = model_from_json(open("emotion_model1.json", "r").read())
        model.load_weights('model.h5')
        prediction = model.predict(image)
        label_dict = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}
        prediction = list(prediction[0])
        img_index = prediction.index(max(prediction))
        final_prediction=label_dict[img_index]
        imagePath = 'static/cropped.jpg'

        optimizer.run(imagePath)
        scene=a3d.Scene.from_file("static/output/mesh0.obj")
        scene.save("static/output/result.glTF")
        return render_template('predict.html', data=final_prediction)


@app.route('/contact')
def main2():
    return render_template('contact.html')

@app.route('/about')
def main3():
    return render_template('about.html')

if __name__ == "__main__":
    app.run()