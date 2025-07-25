# Employee Salary Predictor

This project is a Streamlit web application that predicts an employee's annual salary based on their years at the company and a job performance rating. It allows users to select from different machine learning models and get instant salary predictions.

## Features

* **Interactive Salary Prediction:** Get estimated annual salaries based on user input.
* **Multiple Model Selection:** Choose between Linear Regression, Random Forest, and Gradient Boosting models for prediction.
* **Pre-calculated Model Performance:** View key performance metrics (MAE, RMSE, R²) of the trained models on an unseen test set.
* **User-Friendly Interface:** Built with Streamlit for an intuitive web experience.

## 🛠️ Technologies Used

* **Python**
* **Streamlit:** For building the interactive web application.
* **scikit-learn:** For machine learning model training and evaluation.
* **pandas:** For data manipulation and handling.
* **numpy:** For numerical operations.
* **joblib:** For saving and loading trained machine learning models.
* **matplotlib (optional):** Although not directly used for plotting in the final Streamlit app, it's often part of the data analysis workflow.

##  Project Structure
├── vid/
│   ├── app.py
│   ├── Employees.xlsx
│   ├── linearmodel.pkl
│   ├── randomforest_model.pkl
│   ├── gradientboosting_model.pkl
│   └── Analysis_modelling.ipynb
└── README.md

* `vid/app.py`: The main Streamlit application code.
* `vid/Employees.xlsx`: The dataset used for training the models.
* `vid/*.pkl`: The saved trained machine learning models (Linear Regression, Random Forest, Gradient Boosting).
* `vid/Analysis_modelling.ipynb`: The Jupyter Notebook containing the data analysis, model training, and evaluation steps.
* `README.md`: This file.

##  Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```
2.  **Navigate to the `vid` directory:**
    ```bash
    cd vid
    ```
3.  **Install the required Python packages:**
    It's highly recommended to use a virtual environment.
    ```bash
    # Create a virtual environment (if you don't have one)
    python -m venv venv

    # Activate the virtual environment
    # On Windows:
    # .\venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate

    # Install dependencies
    pip install streamlit pandas numpy scikit-learn joblib openpyxl
    ```
    *(Note: `openpyxl` is needed by pandas to read `.xlsx` files.)*

## Usage

1.  **Ensure all necessary files are in the `vid` directory:**
    Make sure `app.py`, `Employees.xlsx`, and all three `.pkl` model files (`linearmodel.pkl`, `randomforest_model.pkl`, `gradientboosting_model.pkl`) are in the same `vid` folder.
    (These `.pkl` files are generated by running `Analysis_modelling.ipynb`).

2.  **Run the Streamlit application:**
    While inside the `vid` directory in your terminal (or from the parent directory by specifying `vid/app.py`):
    ```bash
    streamlit run app.py
    ```
    Alternatively, if `streamlit` is not directly in your system's PATH, you can use:
    ```bash
    python -m streamlit run app.py
    ```

3.  **Access the App:**
    Your default web browser will automatically open to `http://localhost:8501` (or a similar local address) where you can interact with the Salary Prediction App.

## 📊 Model Performance (Pre-calculated)

The models were evaluated on a test set during development, and their performance metrics are as follows:

| Model             | MAE     | RMSE    | R² Score |
| :---------------- | :------ | :------ | :------- |
| Linear Regression | 8172.23 | 9330.78 | -0.002   |
| Gradient Boosting | 8257.70 | 9479.25 | -0.034   |
| Random Forest     | 8290.66 | 9542.41 | -0.048   |

*(Note: Lower MAE and RMSE indicate better performance, while higher R² Score indicates better fit. An R² score close to 0 or negative for Linear Regression/Gradient Boosting/Random Forest in this context might indicate that the chosen features explain very little of the variance in the target variable, or the model is performing worse than a simple mean baseline. This could be a point for further data analysis or feature engineering.)*

## Built By

**Bhavesh Bhati**
