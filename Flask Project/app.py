from flask import Flask, request, render_template
import pandas as pd
from pickle import load

app = Flask(__name__, static_url_path='/static', template_folder='templates')

# Load the label encoder and model
label_encoder = load(open('label_encoder.pkl', 'rb'))
model = load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request.
    input_data = {
        'age': request.form.get('user_age'),
        'address': request.form.get('user_address'),
        'famsize': request.form.get('user_famsize'),
        'Pstatus': request.form.get('user_Pstatus'),
        'Medu': request.form.get('user_Medu'),
        'Fedu': request.form.get('user_Fedu'),
        'Mjob': request.form.get('user_Mjob'),
        'Fjob': request.form.get('user_Fjob'),
        'reason': request.form.get('user_reason'),
        'guardian': request.form.get('user_guardian'),
        'traveltime': request.form.get('user_traveltime'),
        'studytime': request.form.get('user_studytime'),
        'failures': request.form.get('user_failures'),
        'schoolsup': request.form.get('user_schoolsup'),
        'famsup': request.form.get('user_famsup'),
        'paid': request.form.get('user_paid'),
        'activities': request.form.get('user_activities'),
        'nursery': request.form.get('user_nursery'),
        'higher': request.form.get('user_higher'),
        'internet': request.form.get('user_internet'),
        'romantic': request.form.get('user_romantic'),
        'famrel': request.form.get('user_famrel'),
        'freetime': request.form.get('user_freetime'),
        'goout': request.form.get('user_goout'),
        'Dalc': request.form.get('user_Dalc'),
        'Walc': request.form.get('user_Walc'),
        'health': request.form.get('user_health'),
        'absences': request.form.get('user_absences'),
        'G1': request.form.get('user_G1'),
        'G2': request.form.get('user_G2'),
    }

    # Convert categorical variables using the label encoder
    input_data_encoded = label_encoder.fit_transform(pd.DataFrame({'A': input_data}))


    # Assuming 'input_data_encoded' is a 1D array
    input_data_encoded = input_data_encoded.reshape(1, -1)

    # Make predictions using the model
    prediction = model.predict(input_data_encoded)

    return render_template('result.html', prediction=prediction*5)

if __name__ == '__main__':
    app.run(debug=True)