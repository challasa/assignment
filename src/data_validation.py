import pandas
import pandera.pandas as pa

from pandera.typing import Series
from pandera.errors import SchemaErrors
from typing import Union

from logging_config import setup_logging


validation_logger = setup_logging('data_validation')


class InputDataSchema(pa.DataFrameModel):
    id: Series[int] = pa.Field(unique=True, ge=0)
    stroke: Series[int] = pa.Field(isin=[0, 1], nullable=False)
    age: Series[float] = pa.Field(ge=0, le=100.0)
    hypertension: Series[int] = pa.Field(isin=[0, 1])
    heart_disease: Series[int] = pa.Field(isin=[0, 1])
    avg_glucose_level: Series[float] = pa.Field(ge=55.12)
    bmi: Series[float] = pa.Field(ge=10.0, nullable=True)
    gender: Series[str] = pa.Field(isin=['Male', 'Female', 'Other'])
    ever_married: Series[str] = pa.Field(isin=['Yes', 'No'])
    Residence_type: Series[str] = pa.Field(isin=['Urban', 'Rural'])
    work_type: Series[str] = pa.Field(isin=[
        'Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'
    ])
    smoking_status: Series[str] = pa.Field(isin=[
        'formerly smoked', 'never smoked', 'smokes', 'Unknown'
    ])


class MLReadySchema(pa.DataFrameModel):
    id: Series[float] = pa.Field(unique=True, ge=0)
    stroke: Series[int] = pa.Field(isin=[0, 1], nullable=False)
    age_binned: Series[float] = pa.Field(nullable=False)
    hypertension: Series[float] = pa.Field(isin=[0, 1], nullable = False)
    heart_disease: Series[float] = pa.Field(isin=[0, 1], nullable=False)
    avg_glucose_level: Series[float] = pa.Field(le=4.0,nullable=False)
    bmi_binned: Series[float] = pa.Field(nullable=False)
    gender: Series[float] = pa.Field(nullable=False)
    ever_married: Series[float] = pa.Field(nullable=False)
    residence_type: Series[float] = pa.Field(nullable=False)
    work_type: Series[float] = pa.Field(nullable=False)
    smoking_status: Series[float] = pa.Field(nullable=False)


def validate_dataset(df: pandas.DataFrame,
                     schema: Union[InputDataSchema, MLReadySchema]) -> pandas.DataFrame:
    """Validates a DataFrame and logs errors to a file."""
    try:
        # lazy=True helps collect all errors instead of stopping on the first one
        validated_df = schema.validate(df, lazy=True)
        validation_logger.info("Data validation successful!")
        return validated_df
    except SchemaErrors as err:
        error_message = f"Data validation failed. Total failures: {len(err.failure_cases)}"
        validation_logger.error(error_message)
        # Log the full failure details (the .failure_cases DataFrame)
        validation_logger.error("--- Failure Cases ---\n" + str(err.failure_cases))
        raise err