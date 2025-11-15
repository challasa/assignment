### Stroke Prediction dataset processing pipeline

This repo captures code required to validate and preprocess Stroke Prediction dataset available on Kaggle.


#### **Instructions to run data validation and processing pipeline**

1. Clone the repository
   ```
   git clone https://github.com/challasa/assignment.git
   ```
2. Change into assignment folder, create a venv and activate the venv
   ```
   cd assignment/
   uv venv
   source .venv/bin/activate
   ```
3. Install the dependencies by installing the module
   ```
   uv pip install -e .

   # one can also run `uv pip install -r requirements.txt` or `uv add -r requirements.txt`
   ```
4. Run the python script
   ```
   uv run src/main.py

   # or

   # python src/main.py
   ```

#### Dataset

The data file, available as csv file can be downloaded as zip file from Kaggle or one can use kagglehub module to download. The csv file is available within **data** folder.

#### Data exploration

The data was explored in a Jupyter notebook and using ydata profiler module a report was generated to comprehend univariate and bivariate distributions and relationships. Prpfile report is available within notebooks folder. Since the task is to build a binary classifier, it was decided to implement the following as part of feature engineering steps. These steps can be improved and other approaches can be tried out and these will not necessarily give the best performing model.

    - Impute age and bmi and convert age and bmi into bins
    - Perform ordinal encoding on "ever_married", "Residence_type", "work_type", "gender", "smoking_status" variables
    - Impute and scale avg_glucose_level  

#### Implementation

**Data Validation:**

- In terms of data validation, "pandera" python module is being used to ensure all the instances/rows satisy the checks created. 
- `InputDataSchema` and `ProcessedDataSchema` classes implemented in `src/data_validation.py` can help validate input and processed data. 
- 'Great Expectations' python module is another recommended alternative for carrying out data validation as it has broader capabilities in terms of being able to consume data from several sources, build a comprehensive suite of tests, and also raise triggers/email notifications. If the end environment is in AWS, one can use AWS SNS to notify any data validation errors through email/texts.


**Data Processing:**

- ColumnTransformer from scikit-learn has been used to put together set of pipelines that perform necessary transformation on various columns. `StrokeDataProcessor` class is implemented within `src/data_processor.py`
- Logging has been implemented within `src/logging_config.py` with `config.json` capturing the logger objects, formatters and handlers.