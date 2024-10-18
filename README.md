# Insurance Premium Prediction System

This project provides an API for predicting insurance premiums based on user input using a machine learning model. The model is built with a dataset of insurance data and serves predictions through a Flask web server.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Model Training](#model-training)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   
## Usage

1. Train the machine learning model by running the model.py script
```
python train_model.py
```
This will create a file named insurance_premium_model.pkl containing the trained model.

2. Start the Flask server by running the server.py script:
```
python server.py
```
The server will run on http://127.0.0.1:5000

## API Endpoints

### Predict Insurance Premium

**URL**: `/predict`  
**Method**: `POST`





## Model Training

The model is trained using a dataset from `sample_insurance_data.csv`. The training process involves:

### Data Preprocessing
- Dropping unnecessary columns (`First_Name`, `Last_Name`, `Address`).
- Handling missing values.
- Scaling numerical features and one-hot encoding categorical features.

### Model Selection
- Using a `RandomForestRegressor` for prediction.

### Evaluation
- The model is evaluated using metrics such as Test Score, RMSE, and MAE, which are printed to the console after training.

The trained model is saved to `insurance_premium_model.pkl` for use in the Flask API.

## Contributing

Contributions are welcome! If you have suggestions or improvements, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.

---


