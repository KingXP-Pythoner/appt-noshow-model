# Data Mining/Machine Learning Project: Medical Appointments - No Show

## Goals
1. Given a set of attributes/factors, predict if a person will miss their appointment or not.
2. Determine what factors contribute the most to a person missing their appointment.
3. Compare the performance of the 2 data mining/analysis methods implemented for this project.

## I. Business Understanding

Missed appointments are costly on the medical institutions. Therefore, understanding the factors that cause no-shows are vital in the search for potential solutions to these problems. Having the information about the data set have the following benefits:

1. Hospital can intelligently send more reminders to patients at a higher risk of missing appointments.
2. Understand if the reminder methods (in this case: SMS) are effective or not, and make changes as necessary to the strategies.
3. Inform appointment management/scheduling strategy. (More on the day or more routine appointments?)

## II. Data Understanding
### Dataset:
The dataset contains information about medical appointments and has 14 variables (PatientId, AppointmentID, Gender, DateScheduled, AppointmentDate, Age, Neighborhood, Scholarship, Hypertension, Diabetes, Alcoholism, Handicap, SMSReceived, NoShow).

### Tasks:

Explore the dataset to understand its structure, size, and features.
Check for missing values, outliers, and data types.
Understand the distribution of the target variable (NoShow).
Explore and analyze the relationships between features and the target variable.

## Data Size, Dimensionality, Size, Data types
The dataset provided by [source] has a 110527 x 14 (mxn) dimensionality. We can identify the following columns and their data types (as nominal, ordinal, or continuous):
1. PatientId: nominal
2. AppointmentID: nominal
3. Gender: nominal
4. ScheduledDay: date type
5. AppointmentDay: date type
6. Age: continuous
7. Neighbourhood: nominal
8. Scholarship: nominal
9. Hypertension: nominal
10. Diabetes: nominal
11. Alcoholism: nominal
12. Handcap: nominal
13. SMS_received: nominal
14. No-show: nominal

The dataset has 14 columns or characteristics.

## Dataframe overall information
The dataset has no missing values across all rows and columns.

## Descriptive Statistics
1. Minimum age is -1 which is not possible.
2. Scholarship, Hypertension, Diabetes are binary for all rows. But Handicap has a max value of 4. This could mean this attribute should be binary and these >1 values are errors or it means the number of handicaps the patient had. The description provided from the source via Kaggle states it should be represented as True or False, but the Discussions revealed the attribute is the number of handicaps the patient has.

## Data Cleaning
The goal is to remove anomalies from the data to develop data quality. Since in the descriptive statistics check an anomaly was observed in both the age and handicap columns, data cleaning operation can be performed in these columns. We can also ensure date type columns are converted correctly to datetime.
### Steps:
1. Remove the row will the age = -1. Manual removal is done here as it is simply 1 record with this issue and will not significantly impact the age column in correlation to the target variable for data modeling.
2. Convert scheduled_day and appointment_day columns to datetime.


Check if there are enough rows in the dataset for ROMÃO. If that's the case, the loss is negligible.

## Age Group Distribution
From plotting age distribution on a bar chart, baby (0 years) patients have the most frequency. The distribution is slightly left skewed meaning only a minority sample of the patient population in the dataframe were of the senior/elderly population.

## Comparison of attendance between genders (Male and Female)

The analysis shows that 64.9% of females attended their appointments versus 35.1% of males, and 65.39% of females did not attend versus 34.61% of males. This indicates that while the dataset is skewed towards females, gender here is not a strong predictor of no-show behavior due to the similar percentage distributions across both attendance and no-shows. Consequently, to achieve the project's aim of improving attendance rates, it is crucial to explore other variables such as age, medical conditions, and the impact of SMS reminders, which may provide stronger correlations and insights into patient attendance patterns.

## Comparison of attendance by chronic disease
(Note to self): Get reference for alcoholism definition as a chronic disease.
In exploring the correlation between chronic diseases and appointment attendance, our objective is to understand whether patients with chronic conditions may demonstrate distinct attendance patterns compared to those without such ailments. The analysis unveils a noticeable contrast in attendance rates, with 82.23% of patients with chronic diseases attending appointments versus 79.09% of those without. Conversely, 17.77% of patients with chronic diseases missed appointments, while 20.91% of those without chronic diseases did.

This disparity, a 3.14% difference in attendance rates, although may be thought of being relatively small, slightly suggests that ongoing health management may influence attendance behavior, providing insights for healthcare providers to tailor interventions and enhance appointment adherence across patient groups. However, the scale of this influence may not be determined yet as the data collection occured in a short time period. A longer time frame collection may yield better clarity in understanding this influence. But for the goal of the data exploration and modeling, chronic diseases such as hypertension, diabetes and alcoholism do not show a noteworthy correlation with appointment adherence.

## Attendance comparison based on SMS received
In analyzing the correlation between SMS reception and appointment attendance, our aim is to discern whether patients who receive SMS reminders exhibit different attendance behavior compared to those who don't. The results reveal a notable difference in attendance rates: 83.30% of patients who did not receive an SMS reminder attended their appointments, while 16.70% did not. The discrepancy show that sending SMS reminders actually had an opposite outcome of the expectation that the reminders would improve attendance. However, we need to investigate how same-day appointments contributes to these findings.

## Same day appointments statistics
Roughly 35% of all appointments recorded were same-day appointments. This distribution is significant enough to influence the results gathered earlier. Therefore, it is necessary to filter out same-day appointments as this will be the real test of the impact of the sms campaign.

After filtering out same-day appointments, the new analysis revealed that patients who did not receive an SMS had a show percentage of 70.55% and a no-show percentage of 29.45%. Those who received an SMS showed a slight increase in attendance, with a show percentage of 72.43% and a no-show percentage of 27.57%. This suggests that, for non-same-day appointments, receiving an SMS has a modest positive impact on attendance, improving the show rate by approximately 2% compared to those who did not receive an SMS.

## Attendance comparison by Handicap

Based on the analysis of attendance comparison based on the level of handicap, we observe varying trends. The majority of appointments involve patients with no reported handicap, comprising approximately 97.97% of the dataset. Among these appointments, the no-show rate is 20.24%, indicating a moderate but notable proportion of missed appointments. Interestingly, appointments involving patients with a reported handicap level of 1 or 2 exhibit slightly lower no-show rates compared to those with no reported handicap, suggesting a potential correlation between a mild level of handicap and increased appointment attendance. However, caution is warranted in interpreting these findings due to the relatively small sample sizes of patients with higher levels of handicap (levels 3 and 4), which may not be representative. Further investigation with larger datasets or stratified analyses by handicap severity may provide deeper insights into the relationship between handicap level and appointment attendance.

## Attendance Comparison based on Scholarship status

We observe that the majority of patients without scholarship status attended their appointments, with an attendance rate of 80.19%. Conversely, patients with scholarship status had a slightly lower attendance rate of 76.26%.

## Attendance comparison based on Neighborhood

The variability in the percentages of attendance per neighbourhood shows neighbourhood has a strong effect on attendance, perhaps more than other features explored in this analysis. This may be the factor that contributes most to attendance and this might need to be investigated further, although that is beyond the scope of the data analysis.

## Class Imbalance Investigation
There is a significant imbalance between the classes as over 88k patients attended their appointments versus over 22k missing their appointments. A similar imbalance still appears even after filtering out same-day appointments as it was already known that 35% of the appointments were same-day appointments which majorly were shows (No in no_show class). This occurence must be considered during data modeling. This also means the metric for model quality may not be accuracy and might be other metrics like F1 Score and ROC AUC. Another possible technique that can be implemented could be Random Undersampling.

# Data Modelling
As we proceed with the data modeling stage, two data modeling techniques were chosen for predicting if a patient will miss their appointments or not. The model techniques used are:
1. Logistic Regression Classifier
2. Neural Network Classifier

## 1. Convert categorical features to numerical

## Model Design Thought Process

## Knowledge Summary
This is what we know from the dataset.
1. The target variable is significantly imbalanced with more patients attending vs no shows.
2. The dataset has a mix of categorical and numerical features. But we have converted the categorical features to numerical.
3. The dataset has no missing or duplicate values.
4. We have used frequency encoding for the neighbourhood column which is a high cardinality column.
5. We have normalized the age column.
6. We have evaluated the correlations between the features and the target variable using the Chi-Square test and the Pearson correlation coefficient.
7. From the correlation tests, we can see the top 5 features that are most related to the target variable are: sms_received, hypertension, scholarship, neighbourhood_encoded, and age_scaled.

## Next Steps
1. We will split the dataset into training, validation, and test sets. We will use 70% of the data for training, 15% for validation, and 15% for testing
2. We will need to handle the class imbalance in the training set by using Random undersampling.
3. We will use a Random Forest Classifier to build the model.
4. We will evaluate the model using the validation set.
5. We will fine-tune the model using GridSearchCV.
6. We will evaluate the model using the test set.

### We have neighbourhood column with high cardinality. Using one-hot encoding will increase the number of columns significantly. Therefore, we can use frequency encoding to encode the neighbourhood column.

## Feature Selection

Before modeling, we need to select what features may contribute the most information gain to the model, i.e correlates with the target variables. From the dataset, most of the features are binary, including the target variables. For categorical input features with categorical output/target, a well-known method for determining the correlation is called Chi-Square Test. However, we have converted the neighbourhood and age features to non-binary numerical features. For these columns, we may need to apply a different correlation discovery technique called Pearson Correlation Coefficient.

## Chi Square 
I will select the best 3 features with the highest importance from the results of conducting the Chi Square. This works by choosing the three features with the highest chi values and lowest p values.

# Pearson Correlation Coefficient
As there are only 2 input features,adding to the previously selected 3 features from the Chi-Square test gives 5 features which is not too much for the model. So after visualization, we can add both features as inputs for the models and evaluate the performance.

# Stratified K-Fold Cross Validation on Training Set With Logistic Regression and Random Forest

This stage involved the validation of the base models of Logistic Regression and Random Forest Classifier. These models are widely used in classification problems and compatible with the dataset for this project.

This was implemented via cross_val_score from the sklearn library with a stratified K-Fold of 5 number of splits with shuffling set to true. This enables validating of the models on the training consistently as a simple train, validation, test split may generate inconsistent metrics depending on the split distribution per turn.

**Hyperparameter Tuning with GridSearchCV**

The base logistic regression and random forest models performed poorly especially on metrics for the positive class (No Shows 1). Base Random Forest Classifier performed the worst out of the two, in predicting patient's not showing up which is the main concern for the project. However, we can use GridSearchCV, a method provided by the scikit-learn library that exhaustively runs the base models using all possible combinations of a parameter grid to perform validation. The best model is returned with the optimal hyperparameters.
To perform hyperparameter tuning, I will:
1. Define the hyperparameters to tune for each model
2. Use the training set for 
3. Run GridSearchCV cross-validation for each model on the training set with a K-fold of 5 and store the optimal parameters.
4. Store the optimal parameters and their respective scores. These will be determined using multi-metric scoring based on f1-score, roc auc, and accuracy.

Optimal Parameters and Scores for each model:
lr:
Optimal Parameters: {'C': 1, 'max_iter': 1000, 'penalty': 'l2', 'solver': 'liblinear'}
Score: 0.5962295522051984

rf:
Optimal Parameters: {'max_depth': 40, 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 150}
Score: 0.5905073703760599

With GridSearchCV, it is observed that the best parameters found for the models still yields a poor ROC_AUC score. Therefore even with hyperparameter tuning, the models do not generalize or predict the positive class well.

Optimal Parameters and Scores for each model:
Logistic Regression:
Optimal Parameters: {'C': 10, 'max_iter': 1000, 'penalty': 'l1', 'solver': 'liblinear'}
Score: 0.5955464221008817
Random Forest:
Optimal Parameters: {'max_depth': 40, 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 150}
Score: 0.5888504250473339



Feature engineering is another way of improving the performance of machine learning models, particularly when initial attempts with other methods have not yielded significant improvements. Below are some proposed new features that could potentially enhance the model's ability to predict no-shows:

### Proposed New Features

1. **Days Between Scheduling and Appointment**
   - **Reason**: The time gap between when an appointment is scheduled and when it is actually held could influence the likelihood of a no-show. Longer gaps may lead to more no-shows due to changes in patients' schedules or forgotten appointments.
2. **Previous No-Shows**
   - **Reason**: Patients with a history of no-shows are more likely to miss future appointments. This feature can capture the no-show behavior of patients.


**Handle rows where days_between is negative as appointment day cannot come before the scheduled day.**

**There are a lot of negatives. After inspecting, we can see that the scheduled_day has the time included while the appointment day doesnt. This leads to the scheduled_day being ahead by time. This can be resolved by removing time from the scheduled day.**

**Predict target variable given new features**

This stage involves predicting the classes using the new features added to the dataframe. From the results, a significant improvement was observed after a 5-fold cross validation.
Both models were able to predict No-shows significantly better. This shows a clear positive impact of the derived features via feature engineering. One can attempt to improve this performance even further as although the f1-score for the No show improved for both models, through a round of hyper parameter tuning, the models might improve even more in these metrics.

Logistic Regression Confusion Matrix:
          Show  No-show
Show     59375    11064
No-show   2793    15025
Logistic Regression Classification Report:
              precision    recall  f1-score   support

        Show       0.96      0.84      0.90     70439
     No-show       0.58      0.84      0.68     17818

    accuracy                           0.84     88257
   macro avg       0.77      0.84      0.79     88257
weighted avg       0.88      0.84      0.85     88257

Logistic Regression Average ROC AUC Score: 0.9065

Random Forest Classifier Confusion Matrix:
          Show  No-show
Show     59175    11264
No-show   1126    16692
Random Forest Classifier Classification Report:
              precision    recall  f1-score   support

        Show       0.98      0.84      0.91     70439
     No-show       0.60      0.94      0.73     17818

    accuracy                           0.86     88257
   macro avg       0.79      0.89      0.82     88257
weighted avg       0.90      0.86      0.87     88257

Random Forest Classifier Average ROC AUC Score: 0.9289

**Hyper parameter Tuning Using GridSearchCV and Derived Features**

The results with tuning is also greatly improved. With the metric scoring being ROC AUC, the GridSearchCV provided the optimal parameters for Logistic Regression and Random Forest Classifier. However, it is observed the score for Logistic regression was slightly lower with the optimal parameters than the base model. This may be due to the GridSearchCV limitation where implementing random undersampling was not possible. Nevertheless, the ROC AUC score was satisfactory (close to 1). The evaluation of the base and tuned logistic regression models may be done and results compared to see if tuning was impactful.

Optimal Parameters and Scores for each model:
lr:
Optimal Parameters: {'C': 1, 'max_iter': 1000, 'penalty': 'l1', 'solver': 'liblinear'}
Score: 0.896095247038778

rf:
Optimal Parameters: {'max_depth': 40, 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 150}
Score: 0.9401006893521581

**Perform validation using optimal parameters**

To identify the performance difference with and without hyperparameter tuning, I validated the models with the parameters set. However, for logistic regression, the difference was negligible (0.9065 - 0.9079) for the ROC AUC score. This means the model did not benefit much from hyperparameter tuning. Instead, feature engineering contributed the most.

Logistic Regression (Optimal) Confusion Matrix:
          Show  No-show
Show     58655    11784
No-show   2213    15605
Logistic Regression (Optimal) Classification Report:
              precision    recall  f1-score   support

        Show       0.96      0.83      0.89     70439
     No-show       0.57      0.88      0.69     17818

    accuracy                           0.84     88257
   macro avg       0.77      0.85      0.79     88257
weighted avg       0.88      0.84      0.85     88257

Logistic Regression (Optimal) Average ROC AUC Score: 0.9079

For Random Forest Classifier, the results simply observing the numbers improved with tuning when compared to the base model. It was also higher values than the logistic regression model (base and tuned).

Random Forest Classifier (Optimal) Confusion Matrix:
          Show  No-show
Show     59250    11189
No-show    749    17069
Random Forest Classifier (Optimal) Classification Report:
              precision    recall  f1-score   support

        Show       0.99      0.84      0.91     70439
     No-show       0.60      0.96      0.74     17818

    accuracy                           0.86     88257
   macro avg       0.80      0.90      0.82     88257
weighted avg       0.91      0.86      0.87     88257

Random Forest Classifier (Optimal) Average ROC AUC Score: 0.9367

## Model Testing without Random Undersampling

So far, through feature engineering, we have been able to improve the models' performance significantly as they are able to generalize more on the positive and negative class predictions. Also, hyperparameter tuning the model has been validated and showed a modest but positive impact on the model performance. One can safely proceed to fitting these tuned models on the complete training data and test their performance on the testing set which comprises 20% of the entire dataset. To have exhaustive understanding, this stage performs testing without training data random undersampling and with this resampling. The testing set maintained the class imbalance to mimic the real word as much as possible.
Logistic Regression Model:
Classification Report:
              precision    recall  f1-score   support

        Show       0.84      0.95      0.89     17610
     No-show       0.61      0.31      0.41      4455

    accuracy                           0.82     22065
   macro avg       0.73      0.63      0.65     22065
weighted avg       0.80      0.82      0.80     22065

Confusion Matrix:
                Predicted Show  Predicted No-show
Actual Show              16720                890
Actual No-show            3085               1370
ROC-AUC Score:
0.6284900873202974

Random Forest Classifier Model:
Classification Report:
              precision    recall  f1-score   support

        Show       0.94      0.90      0.92     17610
     No-show       0.66      0.78      0.72      4455

    accuracy                           0.87     22065
   macro avg       0.80      0.84      0.82     22065
weighted avg       0.89      0.87      0.88     22065

Confusion Matrix:
                Predicted Show  Predicted No-show
Actual Show              15805               1805
Actual No-show             962               3493
ROC-AUC Score:
0.8407821351887225

# Model Testing With Training set Random Undersampling

Randomly undersampling the training set produced high ROC AUC scores for both models. This showed that class imbalance negatively impacts a model's performance. The results are as follows:

Logistic Regression Model (Random Under Sampling):
Classification Report:
              precision    recall  f1-score   support

        Show       0.96      0.83      0.89     17610
     No-show       0.57      0.87      0.69      4455

    accuracy                           0.84     22065
   macro avg       0.76      0.85      0.79     22065
weighted avg       0.88      0.84      0.85     22065

Confusion Matrix:
                Predicted Show  Predicted No-show
Actual Show              14624               2986
Actual No-show             562               3893
ROC-AUC Score:
0.8521434293722766

Random Forest Classifier Model (Random Under Sampling):
Classification Report:
              precision    recall  f1-score   support

        Show       0.99      0.84      0.91     17610
     No-show       0.60      0.95      0.74      4455

    accuracy                           0.86     22065
   macro avg       0.79      0.90      0.82     22065
weighted avg       0.91      0.86      0.87     22065

Confusion Matrix:
                Predicted Show  Predicted No-show
Actual Show              14775               2835
Actual No-show             208               4247
ROC-AUC Score:
0.8961614058434046

## Model Evaluation

Both fitted models' results have been established to perform well on the testing dataset, confirming their ability to predict the No-show class. However, to decide on the best model, accuracy which is the proportion of items that were classified correctly, alone is not the only metric to consider as ROC AUC score and f1-score better describe the models' performance contrasts a random guessing classifier. For the goal of hospitals, predicting the positive class (patient will miss an appointment) is likely to be more important than the negative class. But in the real world, the negative class is usually the majority. Therefore, this class imbalance reduces the importance of accuracy and emphasizes recall, precision, f1-score and ROC AUC score, especially for the positive class. On these metrics, these models perform well. But the goal of this phase is to determine if the performance difference based on the metric interests mentioned are statistically significant. We can also determine the metrics on a 95% confidence interval.

Logistic Regression metrics with 95% CI:
Accuracy: Mean = 0.8414, 95% CI = (0.8382, 0.8441)
ROC AUC: Mean = 0.9029, 95% CI = (0.8998, 0.9061)
Class 0 Precision: Mean = 0.5718, 95% CI = (0.5606, 0.5817)
Class 0 Recall: Mean = 0.8466, 95% CI = (0.8414, 0.8548)
Class 0 F1 Score: Mean = 0.6826, 95% CI = (0.6731, 0.6906)
Class 1 Precision: Mean = 0.5718, 95% CI = (0.5606, 0.5817)
Class 1 Recall: Mean = 0.8466, 95% CI = (0.8414, 0.8548)
Class 1 F1 Score: Mean = 0.6826, 95% CI = (0.6731, 0.6906)

Random Forest metrics with 95% CI:
Accuracy: Mean = 0.8590, 95% CI = (0.8560, 0.8639)
ROC AUC: Mean = 0.9337, 95% CI = (0.9313, 0.9367)
Class 0 Precision: Mean = 0.5921, 95% CI = (0.5812, 0.6044)
Class 0 Recall: Mean = 0.9515, 95% CI = (0.9477, 0.9548)
Class 0 F1 Score: Mean = 0.7300, 95% CI = (0.7216, 0.7396)
Class 1 Precision: Mean = 0.5921, 95% CI = (0.5812, 0.6044)
Class 1 Recall: Mean = 0.9515, 95% CI = (0.9477, 0.9548)
Class 1 F1 Score: Mean = 0.7300, 95% CI = (0.7216, 0.7396)


Two-sample z-test for accuracy:
Z-statistic: -6.9736
P-value: 0.0000
There is a statistically significant difference in accuracy between the two models.
Model 2 is the winning model for accuracy.
Two-sample z-test for roc_auc:
Z-statistic: -14.6491
P-value: 0.0000
There is a statistically significant difference in roc_auc between the two models.
Model 2 is the winning model for roc_auc.
Two-sample z-test for precision (class 0):
Z-statistic: -2.5335
P-value: 0.0113
There is a statistically significant difference in precision (class 0) between the two models.
Model 2 is the winning model for precision (class 0).
Two-sample z-test for precision (class 1):
Z-statistic: -2.5335
P-value: 0.0113
There is a statistically significant difference in precision (class 1) between the two models.
Model 2 is the winning model for precision (class 1).
Two-sample z-test for recall (class 0):
Z-statistic: -27.1828
P-value: 0.0000
There is a statistically significant difference in recall (class 0) between the two models.
Model 2 is the winning model for recall (class 0).
Two-sample z-test for recall (class 1):
Z-statistic: -27.1828
P-value: 0.0000
There is a statistically significant difference in recall (class 1) between the two models.
Model 2 is the winning model for recall (class 1).
Two-sample z-test for f1_score (class 0):
Z-statistic: -7.4031
P-value: 0.0000
There is a statistically significant difference in f1_score (class 0) between the two models.
Model 2 is the winning model for f1_score (class 0).
Two-sample z-test for f1_score (class 1):
Z-statistic: -7.4031
P-value: 0.0000
There is a statistically significant difference in f1_score (class 1) between the two models.
Model 2 is the winning model for f1_score (class 1).

## Conclusion


