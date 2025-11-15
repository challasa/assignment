import pandas
import pandera.pandas as pa

from pathlib import Path

from data_processor import StrokeDataProcessor
from data_validation import InputDataSchema, validate_dataset, MLReadySchema
from logging_config import setup_logging

logger = setup_logging()

FILES_PATH = Path(__file__).resolve().parents[1] / "data"
INPUT_FILE_NAME = "healthcare-dataset-stroke-data.csv"
OUTPUT_FILE_PREFIX = "stroke_dataset_cleaned"


def main():
    input_df = pandas.read_csv(FILES_PATH / INPUT_FILE_NAME)
    logger.info("-" * 20)
    logger.info(f"Input file read: {input_df.shape}")
    logger.info(f"Here are the columns: {', '.join(input_df.columns)}")
    logger.info("-" * 20)
    logger.info("Validating input data...")
    logger.info("-" * 20)
    validate_dataset(input_df, schema=InputDataSchema)
    logger.info("-" * 20)
    logger.info("Data processing begin...")
    logger.info("-" * 20)
    processor = StrokeDataProcessor(
        df = input_df,
        test_split_fraction=0.2,
        num_impute_strategy='median', 
        catg_impute_strategy='most_frequent' 
    )
    train, test = processor.fit_transform_data()
    train.to_csv(FILES_PATH / f"{OUTPUT_FILE_PREFIX}_train.csv")
    test.to_csv(FILES_PATH / f"{OUTPUT_FILE_PREFIX}_test.csv")
    logger.info("Data processing end...")
    logger.info("-" * 20)
    logger.info("Validating processed train data...")
    validate_dataset(train, schema=MLReadySchema)
    logger.info("Validating processed test data...")
    validate_dataset(test, schema=MLReadySchema)
    

if __name__ == '__main__':
    main()