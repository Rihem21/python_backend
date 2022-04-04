from flask import Flask
from keras.models import load_model
import numpy as np
from flask import Response,request
import json
import io, requests
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)
print("hello rihem")
model = load_model('mnist_new.h5')
@app.route("/hello")
def hello_world():
    # img = request["img"]
    # resultat ,b = predict_digit(img)
    my_response={"response":"done"}
    return Response(json.dumps(my_response), mimetype='application/json')


@app.route("/model_request",methods=["POST"])
def model_request():
    print("in model")
    # req = request.get_json()
    # print(req)
    try:
        # file = req["file"]
        file = request.files['file']
        img = Image.open(file)
        print(img.size) 
        print("before predict")
        res,acc = predict_digit(img)
        print("after predict")
        my_response={"image":"done","resulat":str(res)}

    except Exception as e :
        print(e)
        my_response={"exception":e}


    print(my_response)
    print("after model")
    return Response(json.dumps(my_response), mimetype='application/json')




def predict_digit(img):
    
    #resize image to 28x28 pixels
    img = img.resize((28,28))
    print("resize")
    #convert rgb to grayscale
    img = img.convert('L')
    print("convert")
    img = np.array(img)
    print("np.array")
    #reshaping to support our model input and normalizing
    img = img.reshape(1,28,28,1)
    print("reshape")
    img = img/255.0
    #predicting the class
    res = model.predict([img])[0]
    print("predict")
    return np.argmax(res), max(res)


app.run(host='0.0.0.0', port=80,debug=True)