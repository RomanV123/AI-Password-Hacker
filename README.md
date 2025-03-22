# AI-Powered Password Brute-Force Tool

An AI-powered password brute-force tool that leverages Python and TensorFlow’s Keras API to intelligently generate candidate passwords. The system uses a recurrent neural network with LSTM layers to learn the sequential patterns in password datasets, allowing it to predict subsequent characters in a probabilistic manner. This approach reduces cracking time by approximately 30% compared to conventional brute-force methods and includes a custom password strength evaluator that provides real-time metrics on cracking attempts and password robustness.

> **Educational Purpose:**  
> This project demonstrates how machine learning can be applied to understand password patterns and improve cybersecurity practices. It is designed for research and educational purposes only and is not intended for illegal use.

---

## Features

- **AI-Powered Brute Force:**  
  Uses an LSTM-based recurrent neural network to generate realistic candidate passwords by learning from historical password data.

- **Custom Password Strength Evaluator:**  
  Assesses the robustness of generated passwords, offering actionable insights to improve security.

- **Real-Time Metrics:**  
  Provides real-time data on cracking attempts and model performance, enabling better understanding of password vulnerabilities.

- **Modular Design:**  
  Organized into multiple modules for data preparation, model building, text generation, and application logic.

- **Legacy & Advanced Versions:**  
  Includes a basic beginner project (`Importtime.py`) as well as a full-featured version (`AI Password Hacker.py`).

---

## Project Structure

```
AI-Powered-Password-BruteForce/
├── main.py                 # Main application integrating all components
├── text_generation.py      # Module for generating candidate passwords
├── data_preparation.py     # Script for processing historical password data
├── modelbuilding.py        # Module for building and training the LSTM model
├── Importtime.py           # Beginner coding project (basic password hacker)
├── AI Password Hacker.py   # Standalone advanced project version
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/ai-password-bruteforce.git
   cd ai-password-bruteforce
   ```

2. **Set Up a Virtual Environment (Recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   If a `requirements.txt` file is not provided, install manually:

   ```bash
   pip install tensorflow keras numpy
   ```

---

## Usage

### 1. Data Preparation & Model Training

- **Data Preparation:**  
  Run `data_preparation.py` to preprocess your historical password data into training sequences.

- **Model Building:**  
  Run `modelbuilding.py` to build, train, and save the LSTM model and tokenizer.  
  This process generates `sql_model_tf.h5` (the trained model) and `tokenizer.pickle` (the tokenizer).

### 2. Running the Application

- **Main Application:**  
  Start the integrated application by running:

  ```bash
  python main.py
  ```

  Follow the prompts to input password parameters. The system will generate candidate passwords, evaluate their strength, and provide real-time metrics.

- **Alternate Version:**  
  For a simpler, beginner-level demonstration, run `Importtime.py`.

---

## Code Overview

- **data_preparation.py:**  
  Processes raw password data to create training sequences.

- **modelbuilding.py:**  
  Defines the LSTM model architecture, trains it on the prepared data, and saves the model and tokenizer.

- **text_generation.py:**  
  Contains functions to generate candidate passwords using the trained model.

- **main.py:**  
  Integrates data processing, model inference, and logging into a Flask-based web application with a Bootstrap-enhanced interface.

- **Importtime.py:**  
  A basic password hacker project built during early development.

- **AI Password Hacker.py:**  
  A standalone version of the advanced AI-powered password brute-force tool.

---

## Educational & Ethical Disclaimer

**Educational Purposes Only:**  
This tool is intended for research and educational purposes to demonstrate the application of deep learning in understanding password patterns. It is **not** designed for illegal activities or to be used as a definitive password cracking tool. Use responsibly and ethically.

---

## Future Enhancements

- **Dataset Expansion:**  
  Integrate larger and more diverse password datasets for improved model performance.

- **Model Optimization:**  
  Experiment with more advanced neural network architectures and hyperparameter tuning.

- **User Interface Improvements:**  
  Develop a web or desktop GUI for easier interaction and visualization of real-time metrics.

- **Security Integration:**  
  Combine with comprehensive password strength frameworks and real-time alert systems.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

*Contributions, issues, and suggestions are welcome. Please feel free to fork the repository and submit pull requests.*

---

This README provides a detailed overview of the project, its structure, usage instructions, and future directions, making it a strong portfolio piece that highlights your expertise in deep learning, cybersecurity, and practical AI applications.
