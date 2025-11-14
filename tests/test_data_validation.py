import pytest
import pandas
from unittest.mock import patch
from src.data_validation import validate_dataset, InputDataSchema, ProcessedDataSchema
from pandera.errors import SchemaErrors

@pytest.fixture
def sample_raw_data():
    """Returns a minimal, mostly valid DataFrame with some issues."""
    data = {
        'id': [1, 2, 3, 4],
        'gender': ['Male', 'Female', 'Other', 'Invalid'], # Invalid gender
        'age': [50.0, 78.0, 150.0, 30.0], # Invalid age (150)
        'hypertension': [1, 0, 1, 0],
        'heart_disease': [0, 1, 1, 0],
        'ever_married': ['Yes', 'No', 'Yes', 'Yes'],
        'work_type': ['Private', 'Self-employed', 'Govt_jov', 'children'],
        'Residence_type': ['Urban', 'Rural', 'Urban', 'Urban'],
        'avg_glucose_level': [100.0, 250.0, 90.0, 60.0],
        'bmi': [25.0, 45.0, 35.0, 20.0], 
        'smoking_status': ['smokes', 'never smoked', 'Unknown', 'formerly smoked'],
        'stroke': [0, 1, 1, 0]
    }
    return pandas.DataFrame(data)

@patch("src.data_validation.validation_logger")
def test_validation_failure_raises_exception(mock_logger, sample_raw_data):
    """Tests that validation correctly raises SchemaErrors on bad data."""
    with pytest.raises(SchemaErrors) as excinfo:
        validate_dataset(sample_raw_data, InputDataSchema)

    mock_logger.error.assert_called() # asserts logger is called
    # Check that the error report contains the expected columns
    failure_cases = excinfo.value.failure_cases
    assert 'gender' in failure_cases['column'].values
    assert 'age' in failure_cases['column'].values 