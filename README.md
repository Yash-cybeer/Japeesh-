# Weather Prediction System

ML-powered Weather Prediction using Gradient Boosting. Predicts Temperature, Humidity, Precipitation & Wind Speed with an interactive Gradio GUI.

## Features

- Predict weather for 5 US cities: San Diego, Philadelphia, San Antonio, San Jose, New York
- Interactive Gradio web interface
- Multi-output prediction: Temperature, Humidity, Precipitation, Wind Speed
- Trained on 1 Million weather records
- Gradient Boosting algorithm with MAE: 9.38

## Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jweTD2dBSbMug0o73Dp3wtFojMBIhamy)

Click the badge above to run the Weather Prediction notebook in Google Colab!
## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run in Google Colab
1. Open `Mini_GenCast_Weather_Prediction.ipynb` in Colab
2. Run all cells
3. Use the Gradio interface to make predictions

### Local Usage
```python
from predict import predict_weather

result = predict_weather('San Diego', '2024-06-15 14:00:00')
print(f"Temperature: {result['Temperature_C']}C")
print(f"Humidity: {result['Humidity_pct']}%")
```

## Project Structure

```
Weather-Prediction-ML/
|-- Mini_GenCast_Weather_Prediction.ipynb  # Main notebook
|-- best_weather_model.pkl                  # Trained model
|-- weather_scaler.pkl                      # Data scaler
|-- requirements.txt                        # Dependencies
|-- README.md
|-- LICENSE
```

## Model Performance

| Model | MAE | R2 Score |
|-------|-----|----------|
| Gradient Boosting | 9.38 | 0.0046 |
| Random Forest | 9.38 | 0.0046 |
| Ridge Regression | 9.39 | 0.0001 |

## Technologies Used

- Python 3.x
- Scikit-learn
- Gradio
- Pandas
- NumPy
- Joblib

## Author

**Yash Bhardwaj** - [@Yash-cybeer](https://github.com/Yash-cybeer)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
