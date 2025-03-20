WILL THEY BUY AGAIN?

 Goal
 
The project focuses on primary objectives:

Predict whether customers will place future orders using CNN, RNN, and a hybrid RNN+LSTM model based on demographic data such as age, occupation, monthly income, and family size. 

 Dataset
 
From Kaggle

The dataset consists of 388 entries with 55 columns, providing detailed insights into customer preferences and experiences in online food delivery. Key variables include:

Demographics: Age, gender, marital status, occupation, income, and education levels.
Geographical Data: Latitude, longitude, pin codes.
Service Preferences: Meal and medium preferences (e.g., "Medium (P1)", "Meal (P1)"), ease, convenience, and time-saving benefits.
Delivery Experience: Factors like restaurant choices, payment options, food quality, tracking systems, and road conditions.
Challenges: Common issues such as late deliveries, hygiene concerns, wrong orders, missing items, and delivery delays.
User Feedback: Reviews, Google Maps accuracy, politeness of delivery personnel, and the influence of ratings.
Outcome: Whether this consumer will buy again or not (e.g., "Output").
 Description:
 
Customer Reordering Behavior Prediction: For predicting whether a customer will reorder, CNN, RNN, and RNN+LSTM models were used. These models are trained on the structured dataset containing demographic data and behavioral factors.



 What I had done!
 
Data Preprocessing: -Removed missing and irrelevant data (e.g., 'Nil' reviews). -Tokenized reviews and converted them into sequences suitable for deep learning models.

Exploratory Data Analysis (EDA): -Analyzed the distribution of customer demographics such as age, income, family size etc. -Created visualizations like bar charts and word clouds for reviews to understand sentiment polarity.

Model Implementation for Prediction: -Built CNN, RNN, and RNN+LSTM models to predict customer reordering behavior. -Experimented with different architectures to capture patterns in structured data.

Evaluation and Comparison: -Compared models using accuracy, precision, recall, and F1-score. -Identified the most accurate models for each task.

 Models Implemented

 
For Customer Reordering Prediction: CNN (Convolutional Neural Network): Captures local patterns and relationships in structured data such as demographic information. RNN (Recurrent Neural Network): Captures sequential dependencies between behavioral factors, though it can suffer from vanishing gradients. RNN + LSTM (Hybrid Model): Handles long-term dependencies more effectively than standard RNNs, providing more accurate predictions for reordering behavior.

Libraries Needed

Used for data manipulation and analysis, especially with DataFrames.
numpy

Supports numerical operations, including working with arrays, matrices, and mathematical computations.
scipy

Provides statistical functions and tools for scientific computing.
matplotlib

A plotting library used for creating visualizations like graphs and charts.
seaborn

A data visualization library built on Matplotlib, offering advanced statistical plots.
scikit-learn (sklearn)

A machine learning library providing tools for:
Data preprocessing (e.g., LabelEncoder, StandardScaler)
Model evaluation (e.g., classification_report, accuracy_score)
Train-test data splitting (e.g., train_test_split)
tensorflow

A deep learning framework used to build and train neural networks.
tensorflow.keras

A high-level neural network API within TensorFlow, offering:
Model creation using Sequential
Neural network layers (e.g., Dense, LSTM, GRU, Dropout)
Regularization techniques (e.g., L2 regularizers)
Text processing tools (e.g., Tokenizer, pad_sequences) .

📈 Performance of the Models based on the Accuracy Scores
Model	Accuracy
CNN	90%
RNN	92%
RNN + LSTM	92.3%

 Conclusion
The project successfully demonstrates the potential of deep learning models for both customer behavior prediction and sentiment analysis.

The RNN + LSTM Hybrid model provided the best performance for sentiment analysis, achieving an accuracy of 92.3%. The LSTM model effectively predicted reordering behavior with 92% accuracy. These insights help food delivery companies better understand customer preferences and enhance their service quality.

Signature
Your Name: HARISH.K

GitHub: https://github.com/Harish3fire

LinkedIn: http://www.linkedin.com/in/harish04k
