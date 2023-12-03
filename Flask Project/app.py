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

@app.route('/process_form', methods=['POST'])
def predict():
    # Get the data from the POST request.
    input_data = {
        'age': float(request.form.get('user_age')),
        'address': request.form.get('user_address'),
        'famsize': request.form.get('user_famsize'),
        'Pstatus': request.form.get('user_Pstatus'),
        'Medu': request.form.get('user_Medu'),
        'Fedu': float(request.form.get('user_Fedu')),
        'Mjob': request.form.get('user_Mjob'),
        'Fjob': request.form.get('user_Fjob'),
        'reason': request.form.get('user_reason'),
        'guardian': request.form.get('user_guardian'),
        'traveltime': float(request.form.get('user_traveltime')),
        'studytime': float(request.form.get('user_studytime')),
        'failures': float(request.form.get('user_failures')),
        'schoolsup': request.form.get('user_schoolsup'),
        'famsup': request.form.get('user_famsup'),
        'paid': request.form.get('user_paid'),
        'activities': request.form.get('user_activities'),
        'nursery': request.form.get('user_nursery'),
        'higher': request.form.get('user_higher'),
        'internet': request.form.get('user_internet'),
        'romantic': request.form.get('user_romantic'),
        'famrel': float(request.form.get('user_famrel')),
        'freetime': float(request.form.get('user_freetime')),
        'goout': float(request.form.get('user_goout')),
        'Dalc': float(request.form.get('user_Dalc')),
        'Walc': float(request.form.get('user_Walc')),
        'health': float(request.form.get('user_health')),
        'absences': float(request.form.get('user_absences')),
        'G1': float(request.form.get('user_G1')),
        'G2': float(request.form.get('user_G2')),
    }

    # Convert categorical variables using the label encoder
    input_data_encoded = label_encoder.transform(pd.DataFrame(input_data, index=[0]))

    # Make predictions using the model
    prediction = model.predict(input_data_encoded)

    return render_template('index.html', prediction_text='Predicted G3: {}'.format(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)
