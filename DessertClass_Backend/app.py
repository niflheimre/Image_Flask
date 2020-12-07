from load import *
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
from imageio import imsave, imread
from PIL import Image
import numpy as np
import keras.models
import cv2
import re
import sys
import os
import base64
from model.load import init
from tensorflow.python.keras.backend import set_session
from flask.helpers import url_for
import io
import tensorflow as tf

sys.path.append(os.path.abspath("./model"))

app = Flask(__name__)

app.config['SECRET_KEY'] = 'the quick brown fox jumps over the lazy   dog'
app.config['CORS_HEADERS'] = "Content-Type"
cors = CORS(app, resources={r"/foo": {"origins": "127.0.0.1:8648"}})

global graph, model_mobile, sess, model_soft, class_names
model_mobile, graph, sess = init()  # เรียกจาก load.py

class_names = ['kanom_bua_loi', 'kanom_chan', 'kanom_dok_jok', 'kanom_kai_tao', 'kanom_krok',
               'kanom_phoi_tong', 'kanom_salim', 'kanom_sangkhaya_faktong', 'kanom_tong_yib', 'kanom_tong_yod']


@app.route('/', methods=['POST','GET'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def test():
    
    if(request.method == 'POST'):
        # content_type = 'image/jpeg'
        
        # headers = {'content-type': content_type}

        # img_str = ""
        
        # with open("logo.png", "rb") as file:
        #     img_str = base64.b64encode(file.read())
        
            # send http request with image and receive response
            return jsonify({'No POST': 'Nope :P'})
            
    return "Up :)"


@app.route('/predict',methods=['POST'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def api():

    img = convertImage(request.get_json())

    print(img.shape)
    
    resized_image_2 = cv2.resize(img, (180, 180), interpolation=cv2.INTER_CUBIC)


    rgb_tensor_2 = tf.convert_to_tensor(resized_image_2, dtype=tf.float32)

    #Add dims to rgb_tensor
    rgb_tensor_2 = tf.expand_dims(rgb_tensor_2, 0)


    with graph.as_default():
        set_session(sess)
        pre = model_mobile.predict(rgb_tensor_2,steps=1)
        prob = model_mobile.predict(rgb_tensor_2,steps=1)

        response = {'Class': class_names[pre[0]],
                    'Prob': np.amax(prob)}
        return response


def convertImage(imgData1):
    
    print("convertImage")

    base64_data = re.sub('^data:image/.+;base64,', '', imgData1)
    img64 = base64.b64decode(str(base64_data))
    image = Image.open(io.BytesIO(img64))

    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)


if __name__ == '__main__':
  app.run(debug=True, port=8648)
