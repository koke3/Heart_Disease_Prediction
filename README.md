# Heart Disease Prediction

This project aims to predict the presence of heart disease in patients using machine learning techniques. The model is built using the PyCaret library for easy deployment and efficient model management.

## Table of Contents
- [Project Description](#project-description)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Description

The **Heart Disease Prediction** application is an AI-based tool designed to analyze patient data and provide predictions regarding the likelihood of heart disease. The application utilizes a pre-trained model created using the **PyCaret** library, making the analysis and prediction process straightforward and efficient.

The app features a user-friendly graphical interface built with **Streamlit**, offering an easy and intuitive experience for users. Patients can input their health data, such as age, sex, blood pressure, and cholesterol levels, and the application will provide predictions about their health status based on this data.

## Technologies Used

- Python
- Streamlit
- Pandas
- PyCaret
- Scikit-learn

## Dataset

The dataset used in this project is derived from the UCI Machine Learning Repository. It contains various health attributes of patients, including age, sex, blood pressure, cholesterol levels, and more. The goal is to predict whether a patient has heart disease.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/koke3/Heart_Disease_Prediction.git
   cd Heart_Disease_Prediction
   ```

2. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the Heart Disease Prediction application, follow these steps:

1. **Run the Application**:
   After installing the libraries, you can start the application using the command:
   ```bash
   streamlit run app.py
   ```

2. **Access the Application**:
   Once the command is run, a message will appear in the console displaying the local URL (usually `http://localhost:8501`). Open this address in your web browser to access the application interface.

3. **Input Data**:
   In the application interface, you will find a section dedicated to inputting patient data. Enter the required information such as age, sex, chest pain type, blood pressure, cholesterol levels, and more.

4. **Get Predictions**:
   After entering all the information, click the "Predict" button to receive your health status predictions. The application will display the result along with a simple explanation.

## Contributing

Contributions are welcome! If you have suggestions for improvements or features, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
