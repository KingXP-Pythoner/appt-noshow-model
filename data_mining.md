# Data Mining/Machine Learning Project: Medical Appointments - No Show

## Goals
1. Given a set of attributes/factors, predict if a person will miss their appointment or not.
2. Determine what factors contribute the most to a person missing their appointment.
3. Compare the performance of the 2 data mining/analysis methods implemented for this project.

##I. Business Understanding

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

This is the order of steps that will be followed:
1. Perform feature engineering.
    - Convert categorical features to numerical. This can be done via one hot encoding.
    - Create new features as needed.
    - Perform feature selection.
3. Split dataset into training, validation and testing sets.
4. Design and train the models on the training set and hypertune with validation set.
5. Evaluate the model's performance via accuracy, F1, confusion matrix and ROC AUC results from testing set.

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

### Evaluation of the Logistic Regression Model on the Validation Set

#### 1. Model Performance Metrics

**Accuracy**: 0.63

The accuracy of the model is 63%, which indicates that 63% of the predictions made by the model are correct. However, accuracy alone is not a sufficient metric to evaluate the performance of the model, especially in the presence of class imbalance.

**Confusion Matrix**:
```
                 Predicted Negative  Predicted Positive
Actual Negative                5543                2492
Actual Positive                1483                1235
```

**Classification Report**:
```
              precision    recall  f1-score   support

           0       0.79      0.69      0.74      8035
           1       0.33      0.45      0.38      2718

    accuracy                           0.63     10753
   macro avg       0.56      0.57      0.56     10753
weighted avg       0.67      0.63      0.65     10753
```

**ROC AUC Score**: 0.58

#### 2. Detailed Analysis

**Precision, Recall, and F1-Score**:
- **Precision for No-Show (Class 1)**: 0.33
  - This indicates that when the model predicts a patient will miss an appointment, it is correct only 33% of the time.
- **Recall for No-Show (Class 1)**: 0.45
  - This indicates that the model correctly identifies 45% of the actual missed appointments.
- **F1-Score for No-Show (Class 1)**: 0.38
  - The F1-score is the harmonic mean of precision and recall, providing a balance between the two. A score of 0.38 indicates poor performance in predicting missed appointments.

**ROC AUC Score**: 0.58
- The ROC AUC score of 0.58 indicates that the model has limited ability to distinguish between patients who will attend and those who will not. A score closer to 1 indicates better discrimination, while a score closer to 0.5 suggests random guessing.

**Confusion Matrix**:
- **True Positives (TP)**: 1235 (Patients predicted as No-Show who actually missed)
- **False Positives (FP)**: 2492 (Patients predicted as No-Show who actually attended)
- **True Negatives (TN)**: 5543 (Patients predicted as Show who actually attended)
- **False Negatives (FN)**: 1483 (Patients predicted as Show who actually missed)

The model's confusion matrix reveals that:
- It correctly predicts 1235 out of 2718 actual missed appointments.
- It incorrectly predicts 1483 missed appointments as attended.
- It incorrectly predicts 2492 attended appointments as missed.
- It correctly predicts 5543 out of 8035 actual attended appointments.

#### 3. Relation to Business Understanding and Goals

The primary business goal is to identify patients who are likely to miss their appointments so that interventions (such as reminders) can be targeted to improve attendance rates. Given this goal, the performance of the logistic regression model can be evaluated as follows:

**Recall for No-Show (45%)**:
- Recall is crucial in this context because we want to identify as many no-show patients as possible. A recall of 45% means that more than half of the patients who miss their appointments are not being identified by the model, which is suboptimal for the business goal.

**Precision for No-Show (33%)**:
- Precision is important to ensure that resources (like reminders) are not wasted on patients who are likely to attend. With a precision of 33%, the model is generating a large number of false positives, meaning many patients who are predicted to miss their appointments actually attend. This inefficiency could lead to wasted efforts and resources.

**F1-Score (38%)**:
- The F1-score, which balances precision and recall, is low. This suggests that the overall effectiveness of the model in identifying no-shows while minimizing false predictions is not satisfactory.

**ROC AUC Score (0.58)**:
- An ROC AUC score of 0.58 indicates the model's poor performance in distinguishing between the two classes (show vs. no-show). It is only slightly better than random guessing, highlighting the need for improvement.

#### 4. Recommendations for Improvement

To achieve the business goal of accurately identifying patients who are likely to miss their appointments, the following steps can be taken to improve the model:

1. **Feature Engineering**: Create additional features that may better capture the patterns associated with missed appointments, such as interaction terms or derived features from existing ones (e.g., days between scheduling and appointment).

2. **Addressing Class Imbalance**: While the training set was undersampled, consider using techniques like Synthetic Minority Over-sampling Technique (SMOTE) to generate synthetic samples for the minority class or adjusting class weights in the logistic regression model to give more importance to the minority class.

3. **Model Tuning**: Perform hyperparameter tuning using techniques like GridSearchCV or RandomizedSearchCV to find the optimal parameters for the logistic regression model.

4. **Ensemble Methods**: Explore ensemble methods such as Random Forest, Gradient Boosting, or XGBoost, which might capture more complex patterns in the data.

5. **Threshold Adjustment**: Fine-tune the classification threshold to find a balance that improves recall for the no-show class without drastically reducing precision.

By addressing these areas, the model can be enhanced to better meet the business objective of predicting no-show appointments, thereby enabling more effective interventions and improving overall attendance rates.

Feature engineering is a critical step in improving the performance of machine learning models, particularly when initial attempts with other methods have not yielded significant improvements. Below are some proposed new features that could potentially enhance the model's ability to predict no-shows, along with reasons for their inclusion:

### Proposed New Features

1. **Days Between Scheduling and Appointment**
   - **Reason**: The time gap between when an appointment is scheduled and when it is actually held could influence the likelihood of a no-show. Longer gaps may lead to more no-shows due to changes in patients' schedules or forgotten appointments.
   
2. **Previous No-Shows**
   - **Reason**: Patients with a history of no-shows are more likely to miss future appointments. This feature can capture the no-show behavior of patients.
   ```python
   df['previous_no_shows'] = df.groupby('patient_id')['no_show'].transform('sum')
   ```

3. **Cumulative Appointments**
   - **Reason**: The total number of past appointments a patient has scheduled can provide insights into their reliability and commitment to attending appointments.
   ```python
   df['cumulative_appointments'] = df.groupby('patient_id').cumcount() + 1
   ```


**Handle rows where days_between is negative as appointment day cannot come before the scheduled day.**

**There are a lot of negatives. After inspecting, we can see that the scheduled_day has the time included while the appointment day doesnt. This leads to the scheduled_day being ahead by time. This can be resolved by removing time from the scheduled day.**

### Detailed Evaluation of the Logistic Regression Model

#### Goal and Business Understanding
The goal is to predict patients who are likely to miss their appointments (i.e., `no_show` = 1) so that intervention measures can be taken to improve attendance rates. The key is to minimize missed appointments, which can help improve the efficiency of healthcare services and ensure better utilization of resources.

#### Model Performance Metrics

1. **Accuracy: 0.88**
   - This means that 88% of the predictions made by the model are correct. While this is a high accuracy, it is not the sole metric to consider, especially with class imbalance.

2. **Confusion Matrix:**
   ```
                     Predicted Negative  Predicted Positive
   Actual Negative                6747                1329
   Actual Positive                   0                2677
   ```
   - **True Negatives (TN):** 6747 - Patients who were predicted to attend and did attend.
   - **False Positives (FP):** 1329 - Patients who were predicted to miss but attended.
   - **False Negatives (FN):** 0 - Patients who were predicted to attend but missed.
   - **True Positives (TP):** 2677 - Patients who were predicted to miss and did miss.

3. **Classification Report:**
   ```
                 precision    recall  f1-score   support

              0       1.00      0.84      0.91      8076
              1       0.67      1.00      0.80      2677

       accuracy                           0.88     10753
      macro avg       0.83      0.92      0.86     10753
   weighted avg       0.92      0.88      0.88     10753
   ```
   - **Precision (no_show=0):** 1.00
     - This means that out of all the patients predicted to attend, 100% actually attended.
   - **Recall (no_show=0):** 0.84
     - Out of all the patients who actually attended, 84% were correctly predicted.
   - **Precision (no_show=1):** 0.67
     - Out of all the patients predicted to miss, 67% actually missed.
   - **Recall (no_show=1):** 1.00
     - Out of all the patients who actually missed, 100% were correctly predicted.
   - **F1-score:**
     - A harmonic mean of precision and recall. For no_show=1, it is 0.80, which indicates a good balance.

4. **ROC AUC Score: 0.94**
   - The ROC AUC score of 0.94 indicates excellent discriminatory ability of the model to distinguish between patients who will attend and those who will miss their appointments.

### Interpretation and Insights

1. **High Recall for no_show=1 (Missed Appointments):**
   - The recall for predicting missed appointments is perfect (1.00), meaning the model correctly identifies all patients who will miss their appointments. This is crucial for the business goal since it ensures no potential no-show is missed by the model.

2. **Moderate Precision for no_show=1:**
   - The precision for predicting missed appointments is 0.67, indicating that 33% of the predicted no-shows will actually attend their appointments. This can lead to unnecessary interventions for some patients.

3. **No False Negatives:**
   - There are no false negatives, meaning the model does not incorrectly classify any actual no-show as a show. This is highly desirable because it avoids missing out on patients who need intervention.

4. **Impact on Business:**
   - **High Recall:** Ensures that almost all patients who are likely to miss their appointments are identified, allowing for targeted interventions such as reminders or rescheduling.
   - **Moderate Precision:** While there is a risk of some false positives, the impact is less critical than false negatives in this scenario. However, reducing false positives could lead to more efficient resource allocation.

**After testing with diverse split distributions for the dataset splits, one can observe that the model has definitely improved with the addition of these two new features. The accuracy improved up to 86.6% on average as well as the very important recall. The ROC AUC score of 0.91 shows the model prediction capacity is very distinguishable from random guessing.**

**Random Forest Classifier Model.
Implementing this model without the new features yielded poorer performance than Logistic Regression without the features.**

**Performance is much higher with the new features**

### Evaluation of Model Performance:

**Business Goal:** To predict patient no-shows accurately so that the clinic can take preemptive actions to reduce missed appointments.

**Logistic Regression:**
- **Accuracy:** 87.94%
- **Precision:** 69.43%
- **Recall:** 91.07%
- **F1 Score:** 78.79%
- **ROC AUC:** 0.9465

**Random Forest:**
- **Accuracy:** 88.71%
- **Precision:** 73.78%
- **Recall:** 83.90%
- **F1 Score:** 78.51%
- **ROC AUC:** 0.9450

### Detailed Interpretation:

The logistic regression model achieves a high recall of 91.07%, meaning it effectively identifies most patients who will not show up for their appointments. This high recall is essential in a clinical setting where the goal is to minimize no-shows by predicting and intervening with those likely to miss their appointments. The model’s accuracy and F1 score also indicate good overall performance, with an ROC AUC of 0.9465, signifying strong discriminative ability between patients who will show and those who won't.

On the other hand, the random forest model achieves a slightly higher accuracy of 88.71% and a precision of 73.78%, meaning it is better at reducing false positives—patients predicted to no-show but actually show up. This model provides a balanced approach with strong overall performance and an ROC AUC of 0.9450. The recall of 83.90%, while slightly lower than logistic regression, is still substantial, ensuring a significant number of no-shows are accurately predicted.

### Business Implications:

The logistic regression model’s high recall makes it suitable for scenarios where catching as many no-shows as possible is critical. This approach maximizes the clinic’s ability to intervene with patients likely to miss appointments, improving overall attendance rates. However, the random forest model offers a balanced trade-off between precision and recall, which could be beneficial to minimize unnecessary interventions, reducing potential resource wastage.

Given the statistically significant performance difference, the random forest model’s slight edge in accuracy and precision might be preferred for a balanced and reliable prediction strategy. However, if the priority is to ensure almost all no-shows are captured, logistic regression might be more suitable.
