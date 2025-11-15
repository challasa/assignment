import click
import pandas
import numpy
from pathlib import Path
from data_processor import StrokeDataProcessor 
from data_validation import validate_dataset, InputDataSchema, MLReadySchema 
from logging_config import setup_logging

cli_logger = setup_logging()

# the main command group
@click.group(name='stroke_pipeline')
def stroke_pipeline():
    """Stroke Data Preprocessing and Validation CLI."""
    pass

# 'run' subcommand under the main group
@stroke_pipeline.command('run')
@click.option('--input_file', 
              required=True, 
              type=click.Path(exists=True), 
              help='Path to the raw input CSV file.')
@click.option('--output_dir', 
              default=Path.cwd(), 
              type=click.Path(), 
              help='Directory to save the processed file')
@click.option('--outfile_prefix',
              default='stroke_dataset_cleaned',
              type=str,
              help='Prefix string for the output files')
def run(input_file: str, output_dir: str, outfile_prefix:str):
    """
    Executes the full feature engineering pipeline: 
    imputation, scaling, ordinal encoding, and Pandera validation.
    """
    cli_logger.info("--- Data Engineering Pipeline Started ---")
    cli_logger.info(f"Input File: {input_file}")
    
    # read the data
    try:
        data = pandas.read_csv(input_file)
        cli_logger.info(f"Successfully loaded data with {data.shape[0]} rows and {data.shape[1]} columns.")
        cli_logger.info(f"Here are the columns: {', '.join(data.columns)}")
    except Exception as e:
        cli_logger.error(f"FATAL: Error during data ingestion from {input_file}: {e}")
        return

    # data validation
    cli_logger.info("-" * 20)
    cli_logger.info("Validating input data...")
    cli_logger.info("-" * 20)
    validate_dataset(data, schema=InputDataSchema)

    # feature engineering and processing
    try:
        cli_logger.info("-" * 20)
        cli_logger.info("Data processing begin...")
        cli_logger.info("-" * 20)
        processor = StrokeDataProcessor(
            df = data,
            test_split_fraction=0.2,
            num_impute_strategy='median', 
            catg_impute_strategy='most_frequent' 
        )
        train, test = processor.fit_transform_data()
        output_file_str = Path(output_dir) / f"{outfile_prefix}" 
        train.to_csv(f"{output_file_str}_train.csv")
        test.to_csv(f"{output_file_str}_test.csv")
        cli_logger.info(f"Output Training dataset File: {output_file_str}_train.csv")
        cli_logger.info(f"Output Test dataset File: {output_file_str}_test.csv")
        cli_logger.info("Data processing end...")
        cli_logger.info("-" * 20)
    except Exception as e:
        cli_logger.error(f"FATAL: Error during feature transformation: {e}")
        return  
    
    # Post-Processing Validation using Pandera
    cli_logger.info("Validating processed train data...")
    validate_dataset(train, schema=MLReadySchema)
    cli_logger.info("Validating processed test data...")
    validate_dataset(test, schema=MLReadySchema)

    cli_logger.info("--- Data Engineering Pipeline Finished ---")