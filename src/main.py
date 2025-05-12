from joblib import load
import os
from flask import Flask, request, jsonify
app = Flask(__name__)

clf = None
TfV = None


@app.route('/', methods=['GET'])
def index():
    global clf
    global TfV
    message = request.args.get('message', '')
    error = ''
    predict_proba = ''
    predict = ''

    try:
        if len(message) > 0:
            vectorize_message = TfV.transform([message])
            predict = clf.predict(vectorize_message)[0]
            predict_proba = clf.predict_proba(vectorize_message).tolist()

    except BaseException as inst:
        error = str(type(inst).__name__) + ' ' + str(inst)

    return jsonify(message=message, predict_proba=predict_proba, predict=predict, error=error)

if __name__ == '__main__':
    clf = load('Models/SmapHamModel.joblib')
    TfV = load('Models/TfVModel.joblib')
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=True)

