# # Create abstraction functions
# # import the required libraries
# from re import X
# import pandas as pd
# import numpy as np
# from typing import Any, Dict, List, Literal, Optional, Tuple, Union
# from sklearn.metrics import (
#     confusion_matrix,
#     classification_report,
#     roc_auc_score,
#     roc_curve,
# )
# import matplotlib.pyplot as plt

# # import Random undersampler from imblearn
# from imblearn.under_sampling import RandomUnderSampler
# from sklearn.preprocessing import MinMaxScaler


# def split_df(df: pd.DataFrame, target_col_label: str) -> tuple:
#     """
#     Split the dataset into features and target variables.

#     Args:
#         df (pd.DataFrame): The dataset to be split.

#     Returns:
#     X (pd.DataFrame): The features.
#     y (pd.Series): The target variable.
#     """
#     X = df.drop(columns=[target_col_label], axis=1)
#     y = df[target_col_label]
#     return X, y


# def train_val_test_split(
#     X: pd.DataFrame,
#     selected_cols: List[str],
#     lvl1_test_size: float,
#     lvl2_test_size: float,
#     target_col_label: str,
#     random_state: int = 42,
# ) -> tuple:
#     """
#     Split the dataset into training and testing sets.

#     Args:
#         X (pd.DataFrame): The features.
#         y (pd.Series): The target variable.
#         test_size (float): The proportion of the dataset to include in the test split.
#         random_state (int): Controls the shuffling applied to the data before applying the split.

#     Returns:
#     X_train (pd.DataFrame): The training features.
#     X_val (pd.DataFrame): The validation features.
#     X_test (pd.DataFrame): The testing features.
#     y_train (pd.Series): The training target variable.
#     y_val (pd.Series): The validation target variable.
#     y_test (pd.Series): The testing target variable.

#     """
#     from sklearn.model_selection import train_test_split

#     # Initial stratified split into training and temp set (70% training, 30% temp)
#     train_set, temp_set = train_test_split(
#         X, test_size=lvl1_test_size, stratify=X['no_show'], random_state=random_state
#     )

#     # Stratified split of the temp set into validation and test sets (each 15% of the total data)
#     val_set, test_set = train_test_split(
#         temp_set, test_size=lvl2_test_size, random_state=random_state, stratify=temp_set['no_show']
#     )

#     return (
#         train_set[selected_cols],
#         val_set[selected_cols],
#         test_set[selected_cols],
#         train_set[target_col_label],
#         val_set[target_col_label],
#         test_set[target_col_label],
#     )


# def under_sample(
#     X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42
# ) -> tuple:
#     """
#     Undersample the training data to balance the classes.

#     Args:
#         X_train (pd.DataFrame): The training features.
#         y_train (pd.Series): The training target variable.
#         random_state (int): Controls the shuffling applied to the data before applying the split.

#     Returns:
#     X_train_resampled (pd.DataFrame): The resampled training features.
#     y_train_resampled (pd.Series): The resampled training target variable.
#     """
#     # Instantiate the RandomUnderSampler
#     rus = RandomUnderSampler(random_state=random_state)
#     # Resample the training data
#     X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)
#     return X_train_resampled, y_train_resampled


# def scale_features(
#     X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
# ) -> tuple:
#     """
#     Scale the features using MinMaxScaler.

#     Args:
#         X_train (pd.DataFrame): The training features.
#         X_val (pd.DataFrame): The validation features.
#         X_test (pd.DataFrame): The testing features.

#     Returns:
#     X_train_scaled (pd.DataFrame): The scaled training features.
#     X_val_scaled (pd.DataFrame): The scaled validation features.
#     X_test_scaled (pd.DataFrame): The scaled testing features.
#     """
#     # Instantiate the MinMaxScaler
#     scaler = MinMaxScaler()
#     # Fit the scaler on the training data
#     scaler.fit(X_train)
#     # Transform the training, validation, and testing data
#     X_train_scaled = scaler.transform(X_train)
#     X_val_scaled = scaler.transform(X_val)
#     X_test_scaled = scaler.transform(X_test)
#     cols = X_train.columns
#     return (
#         pd.DataFrame(X_train_scaled, columns=cols),
#         pd.DataFrame(X_val_scaled, columns=cols),
#         pd.DataFrame(X_test_scaled, columns=cols),
#     )


# def apply_frequency_encoding(
#     df: pd.DataFrame,
#     no_fmap_columns_dict: Optional[Dict[str, str]] = None,
#     fmap: Optional[Dict[str, float]] = None,
#     fmap_column_dict: Optional[Dict[str, str]] = None,
# ) -> Tuple[pd.DataFrame, Dict[Any, Union[int, float]]]:
#     """
#     Apply frequency encoding to the specified columns.

#     Args:
#         df (pd.DataFrame): The dataset to be transformed.
#         no_fmap_columns_dict (dict): A dictionary where the keys are the columns to be transformed and the values are the new column names.
#         fmap (dict) (optional): An optional dictionary, if provided, the function will use the values in the dictionary as the frequency map instead of calculating the frequencies. if fmap is provided, fmap_column_dict must also be provided and it can only contain one key-value pair. no_fmap_columns_dict will be ignored if fmap is provided.
#     Returns:
#     df (pd.DataFrame): The dataset with frequency encoding applied.

#     """
#     freq_dict: Dict[Any, Union[int, float]] = {}
#     if fmap != None and fmap_column_dict != None:
#         # If fmap is provided, use the values in the dictionary as the frequency map
#         for old_label, new_label in fmap_column_dict.items():
#             df[new_label] = df[old_label].map(fmap).fillna(0)
#     else:
#         # Calculate the frequencies of the values in the specified columns
#         if no_fmap_columns_dict != None:
#             for old_label, new_label in no_fmap_columns_dict.items():
#                 freq = df[old_label].value_counts(normalize=True)
#                 df[new_label] = df[old_label].map(freq)
#                 freq_dict = freq.to_dict()
#     return df, freq_dict


# class ModelRunner:
#     """
#     Run the model and return the accuracy score, confusion matrix, classification report, and ROC AUC score. This function also plots the ROC AUC curve.

#     Args:
#         model (Any): The classification model.
#         X_train (pd.DataFrame): The training features.
#         X_test (pd.DataFrame): The testing features.
#         y_train (pd.Series): The training target variable.
#         y_test (pd.Series): The testing target variable.
#         X_val (pd.DataFrame): The validation features.
#         y_val (pd.Series): The validation target variable.
#         cls_report_as_dict (bool): If True, the classification report will be returned as a dictionary. If False, the classification report will be returned as a string.

#     Returns:
#     val_accuracy (float): The accuracy score for the validation set.
#     val_confusion_matrix (np.ndarray): The confusion matrix for the validation set.
#     val_classification_report (str | dict): The classification report for the validation set.
#     val_roc_auc_score (float): The ROC AUC score for the validation set.
#     val_fpr (np.ndarray): The false positive rate for the validation set.
#     val_tpr (np.ndarray): The true positive rate for the validation set.
#     val_thresholds (np.ndarray): The thresholds for the validation set.
#     test_accuracy (float): The accuracy score for the testing set.
#     test_confusion_matrix (np.ndarray): The confusion matrix for the testing set.
#     test_classification_report (str | dict): The classification report for the testing set.
#     test_roc_auc_score (float): The ROC AUC score for the testing set.
#     test_fpr (np.ndarray): The false positive rate for the testing set.
#     test_tpr (np.ndarray): The true positive rate for the testing set.
#     test_thresholds (np.ndarray): The thresholds for the testing set.

#     """

#     model: Any
#     # pandas.DataFrame
#     X_train: pd.DataFrame
#     y_train: pd.Series
#     X_val: pd.DataFrame
#     y_val: pd.Series
#     X_test: pd.DataFrame
#     y_test: pd.Series
#     val_y_pred: np.ndarray
#     val_y_pred_proba: np.ndarray
#     val_accuracy: float
#     val_confusion_matrix: np.ndarray
#     val_classification_report: str | dict
#     val_roc_auc_score: float
#     val_fpr: np.ndarray
#     val_tpr: np.ndarray
#     val_thresholds = None
#     test_y_pred: np.ndarray
#     test_y_pred_proba: np.ndarray
#     test_accuracy: float
#     test_confusion_matrix: np.ndarray
#     test_classification_report: str | dict
#     test_roc_auc_score: float
#     test_fpr: np.ndarray
#     test_tpr: np.ndarray
#     test_thresholds = None
#     pick_results: Literal["validation", "test", "all"]
#     plot: bool
#     cls_report_as_dict: bool

#     def __init__(
#         self,
#         model,
#         X_train,
#         X_test,
#         y_train,
#         y_test,
#         X_val,
#         y_val,
#         pick_results: Literal["validation", "test", "all"] = "all",
#         plot: bool = True,
#         cls_report_as_dict: bool = False,
#     ):
#         self.model = model
#         self.X_train = X_train
#         self.X_test = X_test
#         self.y_train = y_train
#         self.y_test = y_test
#         self.X_val = X_val
#         self.y_val = y_val
#         self.pick_results = pick_results
#         self.plot = plot
#         self.cls_report_as_dict = cls_report_as_dict

#     def __run_model__(self):
#         # Fit the model on the training data
#         self.model.fit(self.X_train, self.y_train)
#         # Predict the target values
#         self.val_y_pred = self.model.predict(self.X_val)
#         self.test_y_pred = self.model.predict(self.X_test)
#         # Predict the probabilities
#         self.val_y_pred_proba = self.model.predict_proba(self.X_val)[:, 1]
#         self.test_y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
#         # Calculate the accuracy score
#         self.val_accuracy = self.model.score(self.X_val, self.y_val)
#         self.test_accuracy = self.model.score(self.X_test, self.y_test)
#         # Calculate the confusion matrix
#         self.val_confusion_matrix = confusion_matrix(self.y_val, self.val_y_pred)
#         self.test_confusion_matrix = confusion_matrix(self.y_test, self.test_y_pred)
#         # Calculate the classification report

#         self.val_classification_report = classification_report(
#             self.y_val, self.val_y_pred, output_dict=self.cls_report_as_dict if self.cls_report_as_dict == True else False
#         )
#         self.test_classification_report = classification_report(
#             self.y_test, self.test_y_pred, output_dict=self.cls_report_as_dict if self.cls_report_as_dict == True else False
#         )
#         # Calculate the ROC AUC score
#         self.val_roc_auc_score = float(roc_auc_score(self.y_val, self.val_y_pred_proba))
#         self.test_roc_auc_score = float(
#             roc_auc_score(self.y_test, self.test_y_pred_proba)
#         )
#         # Calculate the ROC curve
#         self.val_fpr, self.val_tpr, self.val_thresholds = roc_curve(
#             self.y_val, self.val_y_pred_proba
#         )
#         self.test_fpr, self.test_tpr, self.test_thresholds = roc_curve(
#             self.y_test, self.test_y_pred_proba
#         )

#     def __plot_roc_curve__(self):
#         if self.pick_results == "validation":
#             plt.plot(
#                 self.val_fpr,
#                 self.val_tpr,
#                 label=f"Validation (AUC = {self.val_roc_auc_score:.2f})",
#             )
#             plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
#         elif self.pick_results == "test":
#             plt.plot(
#                 self.test_fpr,
#                 self.test_tpr,
#                 label=f"Test (AUC = {self.test_roc_auc_score:.2f})",
#             )
#             plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
#         else:
#             for data in [
#                 (self.val_fpr, self.val_tpr, self.val_roc_auc_score, "Validation"),
#                 (self.test_fpr, self.test_tpr, self.test_roc_auc_score, "Test"),
#             ]:
#                 fpr, tpr, roc_auc_score, label = data
#                 plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc_score:.2f})")
#             plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
#         plt.xlabel("False Positive Rate")
#         plt.ylabel("True Positive Rate")
#         plt.legend()
#         plt.show()

#     def __returns_in_dict__(self):
#         # returns the results in a dictionary
#         return {
#             "val_y_pred": self.val_y_pred,
#             "val_y_pred_proba": self.val_y_pred_proba,
#             "val_accuracy": self.val_accuracy,
#             "val_confusion_matrix": self.val_confusion_matrix,
#             "val_classification_report": self.val_classification_report,
#             "val_roc_auc_score": self.val_roc_auc_score,
#             "val_fpr": self.val_fpr,
#             "val_tpr": self.val_tpr,
#             "val_thresholds": self.val_thresholds,
#             "test_y_pred": self.test_y_pred,
#             "test_y_pred_proba": self.test_y_pred_proba,
#             "test_accuracy": self.test_accuracy,
#             "test_confusion_matrix": self.test_confusion_matrix,
#             "test_classification_report": self.test_classification_report,
#             "test_roc_auc_score": self.test_roc_auc_score,
#             "test_fpr": self.test_fpr,
#             "test_tpr": self.test_tpr,
#             "test_thresholds": self.test_thresholds,
#         }

#     def __pretty_print__(self, result_type: Literal["validation", "test", "all"]):
#         # Pretty print the results
#         if result_type == "validation":
#             print(f"Validation Set\n")
#             print(f"Accuracy: {self.val_accuracy:.2f}")
#             # print validation confusion matrix as a table with labels of true positive, false positive, true negative, and false negative
#             print(
#                 f'Confusion Matrix:\n{pd.DataFrame(self.val_confusion_matrix, columns=["Predicted Negative", "Predicted Positive"], index=["Actual Negative", "Actual Positive"])}'
#             )
#             print(f"Classification Report:\n{self.val_classification_report}")
#             print(f"ROC AUC Score: {self.val_roc_auc_score:.2f}")
#         elif result_type == "test":
#             print("Test Set")
#             print(f"Accuracy: {self.test_accuracy:.2f}")
#             print(
#                 f'Confusion Matrix:\n{pd.DataFrame(self.test_confusion_matrix, columns=["Predicted Negative", "Predicted Positive"], index=["Actual Negative", "Actual Positive"])}'
#             )
#             print(f"Classification Report:\n{self.test_classification_report}")
#             print(f"ROC AUC Score: {self.test_roc_auc_score:.2f}")
#         else:
#             print(f"Validation Set\n")
#             print(f"Accuracy: {self.val_accuracy:.2f}")
#             print(
#                 f'Confusion Matrix:\n{pd.DataFrame(self.val_confusion_matrix, columns=["Predicted Negative", "Predicted Positive"], index=["Actual Negative", "Actual Positive"])}'
#             )
#             print(f"Classification Report:\n{self.val_classification_report}")
#             print(f"ROC AUC Score: {self.val_roc_auc_score:.2f}")
#             print("\n")
#             print("Test Set")
#             print(f"Accuracy: {self.test_accuracy:.2f}")
#             print(
#                 f'Confusion Matrix:\n{pd.DataFrame(self.test_confusion_matrix, columns=["Predicted Negative", "Predicted Positive"], index=["Actual Negative", "Actual Positive"])}'
#             )
#             print(f"Classification Report:\n{self.test_classification_report}")
#             print(f"ROC AUC Score: {self.test_roc_auc_score:.2f}")

#     def invoke(self) -> tuple:
#         self.__run_model__()
#         self.__plot_roc_curve__() if self.plot == True else None
#         if self.pick_results == "validation":
#             self.__pretty_print__("validation")
#             # return the results for the validation set as a tuple
#             return (
#                 self.val_accuracy,
#                 self.val_confusion_matrix,
#                 self.val_classification_report,
#                 self.val_roc_auc_score,
#                 self.val_fpr,
#                 self.val_tpr,
#                 self.val_thresholds,
#                 None,
#                 None,
#                 None,
#                 None,
#                 None,
#                 None,
#                 None,
#             )
#         elif self.pick_results == "test":
#             self.__pretty_print__("test")
#             # return the results for the testing set as a tuple
#             return (
#                 self.test_accuracy,
#                 self.test_confusion_matrix,
#                 self.test_classification_report,
#                 self.test_roc_auc_score,
#                 self.test_fpr,
#                 self.test_tpr,
#                 self.test_thresholds,
#                 None,
#                 None,
#                 None,
#                 None,
#                 None,
#                 None,
#                 None,
#             )
#         else:
#             self.__pretty_print__("all")
#             # return the results for the validation and testing set as a tuple
#             return (
#                 self.val_accuracy,
#                 self.val_confusion_matrix,
#                 self.val_classification_report,
#                 self.val_roc_auc_score,
#                 self.val_fpr,
#                 self.val_tpr,
#                 self.val_thresholds,
#                 self.test_accuracy,
#                 self.test_confusion_matrix,
#                 self.test_classification_report,
#                 self.test_roc_auc_score,
#                 self.test_fpr,
#                 self.test_tpr,
#                 self.test_thresholds,
#             )


# # Create a pipeline that will split the dataset, undersample the training data, scale the features, and run the model
# def preprocess_to_modelling_pipeline(
#     df: pd.DataFrame,
#     target_col_label: str,
#     model: Any,
#     selected_cols: List[str],
#     lvl1_test_size: float,
#     lvl2_test_size: float,
#     pick_results: Literal["validation", "test", "all"],
#     random_state: int = 42,
#     plot: bool = True,
#     cls_report_as_dict: bool = False,
# ):
#     """
#     Run the pipeline that will split the dataset, undersample the training data, apply frequency encoding, scale the features, and run the model.

#     Args:
#         df (pd.DataFrame): The dataset to be used.
#         target_col_label (str): The label of the target column.
#         model (Any): The classification model.
#         selected_cols (list): The columns to be used as features.
#         test_size (float): The proportion of the dataset to include in the test split.
#         random_state (int): Controls the shuffling applied to the data before applying the split.

#     Returns:
#     val_accuracy (float): The accuracy score for the validation set.
#     val_confusion_matrix (np.ndarray): The confusion matrix for the validation set.
#     val_classification_report (str | dict): The classification report for the validation set.
#     val_roc_auc_score (float): The ROC AUC score for the validation set.
#     val_fpr (np.ndarray): The false positive rate for the validation set.
#     val_tpr (np.ndarray): The true positive rate for the validation set.
#     val_thresholds (np.ndarray): The thresholds for the validation set.
#     test_accuracy (float): The accuracy score for the testing set.
#     test_confusion_matrix (np.ndarray): The confusion matrix for the testing set.
#     test_classification_report (str | dict): The classification report for the testing set.
#     test_roc_auc_score (float): The ROC AUC score for the testing set.
#     test_fpr (np.ndarray): The false positive rate for the testing set.
#     test_tpr (np.ndarray): The true positive rate for the testing set.
#     test_thresholds (np.ndarray): The thresholds for the testing set.
#     """
#     # Split the dataset into features and target variables
#     X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
#         df,
#         selected_cols,
#         lvl1_test_size,
#         lvl2_test_size,
#         target_col_label,
#         random_state,
#     )

#     # Encode the neighbourhood column using frequency encoding for the training set
#     X_train, train_fmap = apply_frequency_encoding(
#         X_train, no_fmap_columns_dict={"neighbourhood": "neighbourhood_freq"}
#     )

#     # Encode the neighbourhood column using the frequency map from the training set for the validation set and testing set
#     X_val, _ = apply_frequency_encoding(
#         X_val, fmap_column_dict={"neighbourhood": "neighbourhood_freq"}, fmap=train_fmap
#     )
#     X_test, _ = apply_frequency_encoding(
#         X_test,
#         fmap_column_dict={"neighbourhood": "neighbourhood_freq"},
#         fmap=train_fmap,
#     )
#     # Drop the original neighbourhood and age column from the training, validation, and testing sets
#     for _df in [X_train, X_val, X_test]:
#         _df.drop(columns=["neighbourhood"], axis=1, inplace=True)
#         # _df.drop(columns=['age'], axis=1, inplace=True)

#     # Scale the features
#     X_train_scaled, X_val_scaled, X_test_scaled = scale_features(X_train, X_val, X_test)

#     # Undersample the training data
#     X_train_resampled, y_train_resampled = under_sample(
#         X_train_scaled, y_train, random_state
#     )

#     # Run the model
#     model_runner = ModelRunner(
#         model,
#         X_train_resampled,
#         X_test_scaled,
#         y_train_resampled,
#         y_test,
#         X_val_scaled,
#         y_val,
#         pick_results,
#         plot,
#         cls_report_as_dict,
#     )
#     return model_runner.invoke()


# def preprocessing_pipeline(df:pd.DataFrame, selected_cols:List[str], target_col_label:str)->Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#     """
#     This function will take the dataset, apply frequency encoding to the neighbourhood column, and drop the original neighbourhood column, scale the features, and return the transformed dataset.
    
#     Args:
#     df (pd.DataFrame): The dataset to be transformed.
#     selected_cols (list): The columns to be used as features.
#     target_col_label (str): The label of the target column.
    
#     Returns:
#     X_train_scaled (pd.DataFrame): The scaled features.
#     """
#     from sklearn.model_selection import train_test_split
    
#     X = df[selected_cols]
#     y = df[target_col_label]
#     # Split into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y, random_state=42)
#     X_train, fmap = apply_frequency_encoding(X_train, no_fmap_columns_dict={"neighbourhood": "neighbourhood_freq"})
#     # Apply frequency encoding to the neighbourhood column for the testing set
#     X_test, _ = apply_frequency_encoding(X_test, fmap_column_dict={"neighbourhood": "neighbourhood_freq"}, fmap=fmap)
#     X.drop(columns=["neighbourhood"], axis=1, inplace=True)
#     X_train.drop(columns=["neighbourhood"], axis=1, inplace=True)
#     X_test.drop(columns=["neighbourhood"], axis=1, inplace=True)
#     X_train, _, X_test = scale_features(X_train, X, X_test)
    
#     return (pd.DataFrame(X_train, columns=X.columns), pd.DataFrame(X_test, columns=X.columns), pd.DataFrame(y_train, columns=[target_col_label]), pd.DataFrame(y_test, columns=[target_col_label]), pd.DataFrame(y, columns=[target_col_label]))

# from sklearn.model_selection import cross_val_score, StratifiedKFold
# from scipy.stats import ttest_rel

# from utility.utils import preprocessing_pipeline

# # Define evaluation metrics
# scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
# n_splits = 10
# # Choose cross-validation method
# k_folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# # Select models
# models = {
#     'Logistic Regression': LogisticRegression(C=10, random_state=42, max_iter=2500),
#     'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
# }
# X_train, X_test, y_train, y_test, _ = preprocessing_pipeline(df=dfm, selected_cols=['age', 'scholarship', 'hypertension', 'sms_received', 'neighbourhood', 'days_between', 'previous_no_shows', 'cumulative_appointments'], target_col_label='no_show')

# print(X_train.shape, y_train.shape)
# print(X_train.head())
# print(y_train.head())
# # Perform cross-validation and record results
# results = {}
# for model_name, model in models.items():
#     model_results = {}
#     for metric in scoring:
#         scores = cross_val_score(model, X_train, y_train, cv=k_folds, scoring=metric)
#         model_results[metric] = scores.mean()
#     results[model_name] = model_results

# # Print results
# for model_name, model_result in results.items():
#     print(f"Model: {model_name}")
#     for metric, score in model_result.items():
#         print(f"{metric}: {score}")

# # Perform statistical tests
# metric_of_interest = 'accuracy'
# model1_scores = [results['Logistic Regression'][metric_of_interest] for _ in range(n_splits)]
# model2_scores = [results['Random Forest'][metric_of_interest] for _ in range(n_splits)]
# t_statistic, p_value = ttest_rel(model1_scores, model2_scores)

# # Evaluate significance
# alpha = 0.05
# if p_value < alpha:
#     print("Performance difference is statistically significant.")
# else:
#     print("Performance difference is not statistically significant.")



