# deep-learning-challenge

## Predicting Success
Alphabet Soup, a nonprofit foundation, is on a mission to maximize the impact of its funding by identifying the ventures with the highest likelihood of success. With access to a rich dataset containing information on over 34,000 organizations, this project leverages the power of deep learning to create a binary classification model that predicts the success of funding applicants.

This initiative combines the analytical rigor of data preprocessing, the computational power of neural networks, and the creativity of optimization to deliver actionable insights for Alphabet Soup.

## Results
- [Final code]()

### Data Preprocessing

- Target Variable: The target variable is IS_SUCCESSFUL, which is labeled 1 for organizations that successfully used their funding, and 0 otherwise.
- Feature Variables: All other variables—after dropping irrelevant columns and encoding categorical variables—were used as features to train the model. These included application type, classification, income amount, and more.
- Removed Variables: The EIN and NAME columns were removed because they serve only as identifiers and do not contribute predictive value.

### Compiling, Training, and Evaluating the Model

- Neural Network Architecture
    - Input layer: 100 neurons
    - Hidden layers: 30 and 10 neurons with ReLU and sigmoid activations
    - Output layer: 1 neuron with sigmoid activation
This structure was selected to balance complexity and performance, while minimizing overfitting.

- Model Performance

    - Training Accuracy (Final Epoch): 80.86%
    - Test Accuracy: 78.52%
    - Test Loss: 0.45
    The model came close to the 80% target and maintained consistent performance between training and test datasets.

- Optimization Attempts

    - Removed additional non-contributing columns: AFFILIATION, SPECIAL_CONSIDERATIONS, USE_CASE, ORGANIZATION
    - Filtered for common application types and classifications
    - Experimented with different activation functions (e.g., tanh)
    - Adjusted neuron count and network depth
    - Tuned learning rate, batch size, and number of epochs
These adjustments helped fine-tune the model to achieve higher generalization accuracy.

### Summary and Recommendation
After multiple iterations, the deep learning model achieved a strong 79.42% accuracy on unseen test data. This suggests that the model is fairly effective at identifying organizations likely to use funds successfully.

### Alternative Model Suggestion
A Random Forest Classifier previously tested in the project produced an accuracy of approximately 76.6%, which was slightly lower than the neural network. However, it requires less tuning and can be more interpretable for decision-making.

Why Consider Random Forest?
- Performs well with mixed-type tabular data
- Robust to noise and outliers
- Easier to interpret using feature importance metrics
- Faster to train and less sensitive to parameter tuning
- For quick iteration or deployment in a low-resource environment, Random Forest could be a strong backup.

---

## Project Goals
The primary objective is to develop and refine a deep learning model that accurately predicts whether an organization funded by Alphabet Soup will succeed. To achieve this, the project emphasizes:
- Data preprocessing to prepare and transform the dataset for machine learning.
- Designing and training a neural network tailored for binary classification.
- Employing optimization strategies to enhance model accuracy beyond the 75% target threshold.

## Dataset and Features
The dataset includes key metadata about organizations, such as:
- **APPLICATION_TYPE**: Type of application submitted.
- **AFFILIATION**: Sector of industry affiliation.
- **CLASSIFICATION**: Government organization classification.
- **USE_CASE**: Purpose of funding.
- **INCOME_AMT**: Income classification.
- **ASK_AMT**: Requested funding amount.
- **IS_SUCCESSFUL**: Indicator of funding success.

Columns such as **EIN** and **NAME**, which do not contribute to the predictive analysis, are excluded during preprocessing.

## Methodology
### 1. Data Preprocessing
The first stage involves cleaning and transforming the data:
- Dropping irrelevant columns like **EIN** and **NAME**.
- Grouping low-frequency categorical values into an "Other" category to reduce noise.
- Encoding categorical variables using one-hot encoding.
- Splitting the dataset into training and testing sets for model evaluation.
- Standardizing numerical data with `StandardScaler` to improve model performance.

### 2. Building the Neural Network
Using TensorFlow and Keras, a binary classification neural network is designed with the following structure:
- Input layer tailored to the dataset’s features.
- Hidden layers with optimized neuron counts and activation functions.
- Output layer with an activation function suitable for binary classification.

The model is compiled, trained, and evaluated to calculate its accuracy and loss. Weights are saved periodically during training to preserve progress.

### 3. Model Optimization
To surpass the 75% accuracy target, various strategies are explored, including:
- Adjusting the dataset by refining feature selection and binning techniques.
- Experimenting with additional neurons and layers in the network.
- Testing alternative activation functions and varying the number of training epochs.

Multiple iterations of these adjustments are conducted, and results are saved for comparison and analysis.