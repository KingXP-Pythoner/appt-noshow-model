# Data Mining/Machine Learning Project: Medical Appointments - No Show

## Abstract

Missed medical appointments pose a significant challenge to healthcare institutions, leading to inefficient use of resources and potential negative impacts on patient health outcomes. This project aims to develop predictive models to identify patients at risk of missing their appointments. Using a dataset containing medical appointment records, we analyze various factors contributing to no-show behavior and evaluate the effectiveness of logistic regression and random forest classifiers in predicting missed appointments. Our findings indicate that specific factors, such as SMS reminders and the time gap between scheduling and the appointment, significantly influence attendance rates. By incorporating these insights, healthcare providers can improve intervention strategies to reduce missed appointments and enhance service efficiency.

## I. Introduction

Missed medical appointments are a pervasive issue in healthcare systems worldwide, resulting in wasted resources and suboptimal patient care. Understanding the underlying factors that contribute to no-shows is critical for developing effective strategies to mitigate this problem. This project leverages data mining and machine learning techniques to predict whether a patient will attend their scheduled appointment. The primary objectives are to identify key factors influencing no-show behavior and compare the performance of logistic regression and random forest classifiers in predicting appointment attendance.

## II. Business Understanding

Healthcare institutions incur significant costs due to missed appointments, which can disrupt schedules and reduce the efficiency of medical services. By accurately predicting no-shows, hospitals can take proactive measures, such as sending reminders or rescheduling appointments, to improve attendance rates. Understanding the effectiveness of these interventions, particularly SMS reminders, can further inform appointment management strategies. The goals of this project are to predict no-shows based on various attributes and to identify the most influential factors contributing to missed appointments.

## III. Data Understanding

### Dataset Description

The dataset comprises 110,527 records of medical appointments with 14 attributes: PatientId, AppointmentID, Gender, ScheduledDay, AppointmentDay, Age, Neighborhood, Scholarship, Hypertension, Diabetes, Alcoholism, Handicap, SMSReceived, and NoShow. The target variable is NoShow, indicating whether a patient attended their appointment.

### Initial Data Analysis

1. **Data Types and Structure:** The dataset includes both nominal/categorical and discrete/continuous variables. Nominal variables include PatientId, AppointmentID, Gender, Neighborhood, Scholarship, Hypertension, Diabetes, Alcoholism, Handicap, SMSReceived, and NoShow. Continuous variables include Age. ScheduledDay and AppointmentDay are date types.
2. **Missing Values:** There are no missing values in the dataset.
3. **Descriptive Statistics:** Initial analysis revealed anomalies, such as an age value of -1 and inconsistent values in the Handicap column, which required data cleaning.

## IV. Data Cleaning

Data cleaning involved the following steps:
1. **Removing Anomalies:** The record with an age of -1 was removed.
2. **Converting Date Types:** The DateScheduled and AppointmentDate columns were converted to datetime format to facilitate time-related calculations.
3. **Handling Inconsistent Values:** Standardizing the interpretation of the Handicap attribute by converting it into binary (handicapped vs. not handicapped) for simplicity.

## V. Exploratory Data Analysis

### Age Distribution

The age distribution revealed a higher frequency of appointments for babies (0 years old), with a left-skewed distribution indicating fewer elderly patients.

### Gender Comparison

Females constituted 64.9% of the dataset, with attendance and no-show rates similar across genders, suggesting that gender is not a strong predictor of no-show behavior.

### Chronic Diseases

Patients with chronic diseases (hypertension, diabetes, alcoholism) showed slightly higher attendance rates. However, the difference was marginal, indicating that chronic diseases do not significantly affect no-show rates.

### SMS Reminders

Initial analysis suggested that patients who did not receive SMS reminders had higher attendance rates. However, filtering out same-day appointments showed that SMS reminders slightly improved attendance for non-same-day appointments, highlighting their modest positive impact.

### Handicap Levels

Patients with mild handicaps (levels 1 or 2) had slightly lower no-show rates compared to those with no handicaps, but the small sample sizes for higher handicap levels warrant cautious interpretation.

### Scholarship Status

Patients without scholarship status had a higher attendance rate (80.19%) compared to those with scholarship status (76.26%), suggesting a potential socioeconomic influence on appointment adherence.

### Neighborhood

Neighborhood had a significant impact on attendance rates, with variability in attendance percentages across different neighborhoods, indicating it as a strong predictor of no-show behavior.

### Class Imbalance

The dataset exhibited a significant class imbalance, with a majority of patients attending their appointments. This imbalance necessitated the use of metrics like F1 Score and ROC AUC for model evaluation, rather than accuracy alone.

## VI. Data Modeling

### Feature Engineering

Feature engineering included converting categorical variables to numerical using one-hot encoding and creating new features, such as the number of days between scheduling and the appointment. 

1. **One-Hot Encoding:** Categorical variables such as Gender, Neighborhood, and Scholarship were transformed into binary columns.
2. **Date Differences:** A new feature was created to represent the number of days between the DateScheduled and the AppointmentDate.

### Model Design

Two machine learning models were implemented:
1. **Logistic Regression Classifier**
2. **Random Forest Classifier**

### Model Training and Validation

The dataset was split into training and testing sets using an 80-20 split. Both models were trained on the training set and validated using the testing set. Cross-validation was used to ensure the robustness of the models.

### Model Evaluation Metrics

Models were evaluated using accuracy, F1 score, confusion matrix, and ROC AUC. These metrics were chosen to account for the class imbalance and provide a comprehensive assessment of model performance.

## VII. Results and Discussion

### Logistic Regression Model

The logistic regression model achieved:
- **Accuracy:** 87.94%
- **Precision:** 69.43%
- **Recall:** 91.07%
- **F1 Score:** 78.79%
- **ROC AUC:** 0.9465

### Random Forest Model

The random forest model achieved:
- **Accuracy:** 88.71%
- **Precision:** 73.78%
- **Recall:** 83.90%
- **F1 Score:** 78.51%
- **ROC AUC:** 0.9450

### Analysis

Both models demonstrated strong performance, with the logistic regression model achieving higher recall, making it more suitable for identifying no-shows. The random forest model, while slightly better in accuracy and precision, showed a balanced trade-off between precision and recall.

### Feature Importance

In the random forest model, feature importance analysis indicated that the number of days between scheduling and the appointment, SMS reminders, and neighborhood were among the most significant predictors of no-shows.

### Business Implications

High recall in the logistic regression model ensures most no-shows are identified, allowing for effective interventions. The moderate precision indicates some false positives, which could lead to unnecessary interventions but is preferable to missing actual no-shows. The random forest model provides a balanced approach, reducing false positives while maintaining good recall.

## VIII. Conclusion

This project demonstrates the application of logistic regression and random forest classifiers to predict medical appointment no-shows. Feature engineering and handling class imbalance significantly improved model performance. The findings highlight the importance of factors such as SMS reminders and the time gap between scheduling and the appointment. By leveraging these insights, healthcare providers can implement targeted interventions to reduce missed appointments, thereby enhancing operational efficiency and patient care outcomes. Future work could explore additional features and advanced ensemble methods to further improve prediction accuracy and robustness.

## IX. References

- [Source of the dataset]
- Relevant academic papers and articles on predicting medical appointment no-shows
- Documentation and tutorials on logistic regression and random forest classifiers

---

This structured report provides a comprehensive overview of the project, detailing the business understanding, data analysis, modeling process, results, and implications. It highlights the significance of the findings and offers actionable insights for healthcare providers.