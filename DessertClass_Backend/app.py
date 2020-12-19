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
# from test import init as pred
from tensorflow.python.keras.backend import set_session
from flask.helpers import url_for
import io
import tensorflow as tf
from tensorflow.python.keras import backend as K
tf.compat.v1.disable_v2_behavior()
sys.path.append(os.path.abspath("./model"))

app = Flask(__name__)

app.config['SECRET_KEY'] = 'the quick brown fox jumps over the lazy   dog'
app.config['CORS_HEADERS'] = "Content-Type"
cors = CORS(app, resources={r"/foo": {"origins": "127.0.0.1:8648"}})

K.clear_session()
global graph, model_mobile, sess, class_names

model_mobile,graph,sess = init()  # เรียกจาก load.py

# model_mobile = keras.models.load_model('model_moblienet_v2.h5', custom_objects={
#     'f1_m': f1_m, 'precision_m': precision_m, 'recall_m': recall_m})

graph = tf.compat.v1.get_default_graph()

class_names = ['kanom bua loi', 'kanom chan', 'kanom dok jok', 'kanom kai tao', 'kanom krok',
               'kanom phoi tong', 'kanom salim', 'kanom sangkhaya faktong', 'kanom tong yib', 'kanom tong yod']


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

    # print(img.shape)
    
    resized_image_2 = cv2.resize(img, (160, 160), interpolation=cv2.INTER_CUBIC)
    # print(rgb_tensor_2.shape)
    # with graph.as_default():
    #     model_mobile.compile()
    #     model_mobile.fit()

    with graph.as_default():
    # #     print("Graph init")
        set_session(sess)
        rgb_tensor_2 = tf.compat.v1.convert_to_tensor(
            resized_image_2, dtype=tf.float32)

        #Add dims to rgb_tensor
        rgb_tensor_2 = tf.compat.v1.expand_dims(rgb_tensor_2, 0)
        
        model_mobile._make_predict_function()
        pre = model_mobile.predict(rgb_tensor_2,steps=1)
        
        name, val = tellclass(pre)

        response = {'class': name,
                    'value': str(val)[:4]}
        # response = {'Class': class_names[pre]}
        # K.clear_session()
        return jsonify(response)


def tellclass(arr):
    return class_names[np.argmax(arr)],np.max(arr)


def convertImage(imgData1):
    
    print("convertImage")

    base64_data = re.sub('^data:image/.+;base64,', '', imgData1)
    img64 = base64.b64decode(str(base64_data))
    image = Image.open(io.BytesIO(img64))

    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)


if __name__ == '__main__':
  app.run(debug=True, port=8648)
