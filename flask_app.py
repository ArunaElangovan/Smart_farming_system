from flask import Flask,request,jsonify
import numpy as np
import pickle
model = pickle.load(open('/Users/aelangovan/CS688_Smart_Farming_AI_model/model.pkl','rb'))
app = Flask(__name__)
@app.route('/')
def index():
    return "Hello world"
@app.route('/predict',methods=['POST'])
def predict():
    n = request.form.get('n')
    p = request.form.get('p')
    k = request.form.get('k')
    temp = request.form.get('temp')
    humidity = request.form.get('humidity')
    ph = request.form.get('ph')
    input_query = np.array([[n,p,k,temp,humidity,ph]])
    result = model.predict(input_query)[0]
    return jsonify({'placement':str(result)})
if __name__ == '__main__':
    app.run(debug=True)