from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
from datetime import datetime
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load the trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class HealthMetric(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    glucose = db.Column(db.Float, nullable=False)
    insulin = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    age = db.Column(db.Integer, nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)


# Initialize the database
with app.app_context():
    db.create_all()

# Routes

@app.route('/')
def home():
    """
    Renders the homepage.
    """
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """
    Handles user registration. Redirects to login page upon success.
    """
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        # Check if email already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return render_template('register.html', message="Email already registered.")

        # Hash the password and save the user
        hashed_password = generate_password_hash(password, method='sha256')
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """
    Handles user login. Redirects to prediction page upon success.
    """
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Check user credentials
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect(url_for('predict'))
        
        return render_template('login.html', message="Invalid email or password.")

    return render_template('login.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    
    if 'user_id' not in session:
        return redirect(url_for('login'))  # Redirect unauthenticated users

    if request.method == 'POST':
        try:
            # Get input data from the form
            glucose = float(request.form['Glucose Level'])
            insulin = float(request.form['Insulin'])
            bmi = float(request.form['BMI'])
            age = float(request.form['Age'])

            # Predict diabetes
            input_data = np.array([[glucose, insulin, bmi, age]])
            scaled_data = scaler.transform(input_data)
            prediction = model.predict(scaled_data)[0]


            # Generate result message
            result = (
                "You have Diabetes, please consult a doctor."
                if prediction == 1
                else "You don't have Diabetes."
            )

            # Save metrics to the database
            health_metric = HealthMetric(
                user_id=session['user_id'],
                glucose=glucose,
                insulin=insulin,
                bmi=bmi,
                age=age
            )
            db.session.add(health_metric)
            db.session.commit()

            # Show actionable health tips if the prediction indicates diabetes
            show_tips = prediction == 1

            return render_template(
                'predict.html', user=session['username'], result=result, show_tips=show_tips
            )
        except Exception as e:
            return render_template(
                'predict.html',
                user=session['username'],
                result=f"Error occurred: {e}",
            )

    return render_template('predict.html', user=session['username'])


@app.route('/health-tips')
def health_tips():
    """
    Provides actionable health tips for users with diabetes.
    """
    tips = [
        "Maintain a healthy diet rich in vegetables, lean proteins, and whole grains.",
        "Engage in regular physical activity like walking, jogging, or yoga.",
        "Monitor your blood sugar levels regularly.",
        "Stay hydrated and avoid sugary beverages.",
        "Consult with a doctor or dietitian for a personalized care plan.",
    ]
    return render_template('health_tips.html', tips=tips)


@app.route('/progress')
def progress():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # Fetch health metrics for the logged-in user
    metrics = HealthMetric.query.filter_by(user_id=session['user_id']).order_by(HealthMetric.date.asc()).all()

    # Prepare data for visualization
    dates = [metric.date.strftime('%Y-%m-%d') for metric in metrics]
    glucose = [metric.glucose for metric in metrics]
    insulin = [metric.insulin for metric in metrics]
    bmi = [metric.bmi for metric in metrics]

    # Pass data to the template
    return render_template('progress.html', dates=dates, glucose=glucose, insulin=insulin, bmi=bmi)




@app.route('/logout')
def logout():
    
    session.clear()
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
