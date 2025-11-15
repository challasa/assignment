import pandas
import numpy

from pandera.typing.pandas import DataFrame
from typing import Literal, Tuple
from data_validation import InputDataSchema
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from logging_config import setup_logging

proc_logger = setup_logging(logger_name='data_processing')


def create_age_bins(X: numpy.ndarray) -> numpy.ndarray:
    """Creates categorical bins for the "age" feature."""
    age_series = pandas.Series(X.flatten())
    bins = [0, 18, 45, 65, 100]
    labels = ["Child", "YoungAdult", "MiddleAged", "Senior"]
    binned_ages = pandas.cut(age_series, bins=bins, labels=labels, right=False).astype(str).to_numpy().reshape(-1, 1)
    # handle NaNs that might be present after cutting (will be 'nan' string)
    binned_ages[binned_ages == 'nan'] = numpy.nan
    return binned_ages


def create_bmi_bins(X: numpy.ndarray) -> numpy.ndarray:
    """Creates categorical bins for the 'bmi' feature."""
    bmi_series = pandas.Series(X.flatten())
    bins = [0, 18.5, 25, 30, 100]
    labels = ["Underweight", "Healthy", "Overweight", "Obese"]
    binned_bmi = pandas.cut(bmi_series, bins=bins, labels=labels, right=False).astype(str).to_numpy().reshape(-1, 1)
    binned_bmi[binned_bmi == 'nan'] = numpy.nan
    return binned_bmi


class StrokeDataProcessor:
    """
    A class to implement feature engineering for the Stroke Prediction Dataset.
    Methods in the class can help return train and test datasets that are cleaned and 
    are ready for model building step.
    """
    def __init__(self, df: DataFrame[InputDataSchema],
                 test_split_fraction: float,
                 num_impute_strategy: Literal["mean", "median"] = "median",
                 catg_impute_strategy: Literal["constant", "most_frequent"] = "most_frequent",
                 target_column: str = "stroke"
                 ):
        """Constructor method of the class

        Params:
            df: input dataframe that satisfies InputDataSchema
            test_split_fraction: fraction of instances to be retained as test
            num_impute_strategy: strategy to adopt to impute missing values in numerical columns. Defaults to "median".
            catg_impute_strategy: strategy to impute categorical missing values in categorical columns. Defaults to "most_frequent".
            target_column: target variable in the input dataframe Defaults to "stroke"
        """
        self.df = df.copy()
        self.test_split_fraction = test_split_fraction
        self.num_impute_strategy = str(num_impute_strategy)
        self.catg_impute_strategy = catg_impute_strategy
        self.target_column = target_column
        # define column types based on the processing steps
        self._numerical_cols = ["avg_glucose_level"]
        self._bmi_col = ["bmi"]
        self._age_col = ["age"]
        
        # Categorical features that need to undergo OrdinalEncoding
        self._ordinal_cols = ["ever_married", "Residence_type", "work_type", "gender", "smoking_status"]
        self._passthrough_cols = ["id", "hypertension", "heart_disease"]
        self._out_feature_names = [
            "age_binned", "bmi_binned", "ever_married", 
            "residence_type", "work_type", "gender", "smoking_status",
            "avg_glucose_level", "id",
            "hypertension", "heart_disease"
        ]

        # save ColumnTransformer output into transformer variable
        self.transformer: ColumnTransformer = None

    def _create_pipeline(self) -> ColumnTransformer:
        """
        Creates a ColumnTransformer that defines the feature engineering steps.
        """
        # *** Numerical columns handling ***
        numerical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=self.num_impute_strategy)),
            ('scaler', StandardScaler())
        ])

        # *** OrdinalEncoding on categorical columns (Impute and then encode) ***
        ordinal_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=self.catg_impute_strategy, fill_value='missing')),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=numpy.int64))
        ])

        # *** Age & BMI binning & Ordinal Encoding ***
        age_binning_pipeline = Pipeline(steps=[
            ('num_imputer', SimpleImputer(strategy=self.num_impute_strategy)), 
            ('bin_transformer', FunctionTransformer(create_age_bins)),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=numpy.int64))
        ])

        bmi_binning_pipeline = Pipeline(steps=[
            ('num_imputer', SimpleImputer(strategy=self.num_impute_strategy)), 
            ('bin_transformer', FunctionTransformer(create_bmi_bins)),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=numpy.int64))
        ])
        
        # *** Combine all steps using ColumnTransformer ***
        preprocessor = ColumnTransformer(
            transformers=[
                # 1. Binned Features (Ordinal Encoded)
                ('age_pipe', age_binning_pipeline, self._age_col), # age_binned
                ('bmi_pipe', bmi_binning_pipeline, self._bmi_col), # bmi_binned
                
                # 2. Ordinal Categorical Features 
                ('ordinal_pipe', ordinal_pipeline, self._ordinal_cols),
                
                # 3. Numerical Features
                ('num_pipe', numerical_pipeline, self._numerical_cols),
            ],
            remainder='passthrough' 
        )
        proc_logger.info(f"Creating binned features for 'age' and 'bmi'...")
        proc_logger.info(f"Encoding following Ordinal features: {', '.join(self._ordinal_cols)}...")
        proc_logger.info(f"Scaling {', '.join(self._numerical_cols)}...")
        return preprocessor
    
    def fit_transform_data(self, 
                           random_state: int = 2025
    ) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
        """
        Splits data, fits the transformation pipeline on the training set,
        and transforms both the training and test sets.
        
        Returns: (train_df, test_df)
        """
        if self.transformer is None:
            self.transformer = self._create_pipeline()
            
        # 1. Split Data (Crucial to prevent data leakage)
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_split_fraction, 
            random_state=random_state
        )
        
        # 2. Fit and Transform the Training Data
        X_train_processed = self.transformer.fit_transform(X_train)
        
        # 3. Transform the Test Data (No fitting)
        X_test_processed = self.transformer.transform(X_test)
        
        # 4. Convert back to DataFrames using the explicit feature names list
        X_train_processed_df = pandas.DataFrame(X_train_processed, 
                                            columns=self._out_feature_names, 
                                            index=X_train.index)
        
        X_test_processed_df = pandas.DataFrame(X_test_processed, 
                                           columns=self._out_feature_names, 
                                           index=X_test.index)
        
        train_df_processed = pandas.concat([X_train_processed_df, y_train.rename(self.target_column)], axis=1)
        test_df_processed = pandas.concat([X_test_processed_df, y_test.rename(self.target_column)], axis=1)

        proc_logger.info("- Final Pipeline Summary -")
        proc_logger.info(f"Train/Test Split: {len(train_df_processed)} / {len(test_df_processed)}")
        proc_logger.info(f"Numerical Imputer: {self.num_impute_strategy}")
        proc_logger.info(f"Categorical Imputer: {self.catg_impute_strategy}")
        proc_logger.info(f"Final Feature Count (including target): {train_df_processed.shape[1]}")
        proc_logger.info("-" * 25)
        return train_df_processed, test_df_processed