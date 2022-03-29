from flask import Flask
from keras.models import load_model
import numpy as np
from flask import Response
import json
model = load_model('mnist.h5')
app = Flask(__name__)

@app.route("/model")
def hello_world():
    img = request["img"]
    resultat ,b = predict_digit(img)
    my_response={"response":"done"}
    return Response(json.dumps(resultat), mimetype='application/json')






def predict_digit(img):
    #resize image to 28x28 pixels
    img = img.resize((28,28))
    #convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)
    #reshaping to support our model input and normalizing
    img = img.reshape(1,28,28,1)
    img = img/255.0
    #predicting the class
    res = model.predict([img])[0]
    return np.argmax(res), max(res)


app.run(host='0.0.0.0', port=80)