# Smart Traffic AI 🚦

## About the Project
Smart Traffic AI is a machine learning project that I developed to practice Python and machine learning. It predicts traffic levels and estimates the number of vehicles based on different traffic conditions.

I also created a Streamlit dashboard where users can enter traffic details and view predictions.

## Features
- Predicts traffic levels: Low, Medium, or High.
- Estimates the number of vehicles.
- Shows charts to understand traffic patterns.
- Provides an interactive dashboard using Streamlit.
- Uses trained machine learning models for predictions.

## Machine Learning Models
- Random Forest Classifier for traffic level prediction.
- Random Forest Regressor for vehicle count prediction.

## Dataset
The project uses a dataset containing 1,000 rows and 9 columns. It includes information such as date, hour, weather, temperature, road type, vehicle count, and traffic level.

The dataset is created for learning and practice, so the predictions should not be considered real-world traffic forecasts.

## Tools and Technologies
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Streamlit
- Joblib

## How to Run
1. Clone or download this repository.
2. Open the project folder in VS Code.
3. Create and activate a Python virtual environment.
4. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

5. Run the application:

   ```bash
   streamlit run app.py
   ```

## What I Learned
While working on this project, I practiced data preprocessing, training classification and regression models, saving trained models, and building a dashboard with Streamlit.

## Future Improvements
I would like to improve the model performance and try using real-world traffic data in the future.