from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open('model/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    hours = float(request.form['hours'])
    attendance = float(request.form['attendance'])
    sleep = float(request.form['sleep'])

    result = model.predict([[hours, attendance, sleep]])

    if result[0] == 1:
        output = "Good Performance Expected ✅"
    else:
        output = "Needs Improvement ⚠️"

    return render_template('result.html', result=output)

if __name__ == '__main__':
    app.run(debug=True)