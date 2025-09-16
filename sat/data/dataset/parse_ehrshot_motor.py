"""Parse EHRSHOT data with medical code sequences for MOTOR-T processing."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

from dataclasses import dataclass
from logging import DEBUG, ERROR
from pathlib import Path

import pandas as pd
from logdecorator import log_on_end, log_on_error, log_on_start

from sat.utils import logging

logger = logging.get_default_logger()


@dataclass
class EHRSHOTMotor:
    """Parse EHRSHOT medical sequences for MOTOR-T transformer backend."""
    
    source: str  # Path to parquet file with sequences
    processed_dir: str
    name: str
    
    @log_on_start(DEBUG, "Parsing EHRSHOT MOTOR sequences...")
    @log_on_error(
        ERROR,
        "Error parsing EHRSHOT MOTOR data: {e!r}",
        on_exceptions=Exception,
        reraise=True,
    )
    @log_on_end(DEBUG, "done!")
    def __call__(self) -> None:
        """Load and format EHRSHOT sequences for SAT pipeline."""
        
        # Load sequences from parquet
        logger.debug(f"Loading sequences from {self.source}")
        df = pd.read_parquet(self.source)
        logger.debug(f"Loaded {len(df)} patient sequences")
        
        # SAT expects these columns
        # x: text sequences (medical codes)
        # events: multi-event indicators
        # durations: time to events
        # Additional columns preserved: patient_id, split
        
        # The data already has the right column names from prepare_ehrshot_sequences.py
        # Just need to ensure proper formatting
        
        # Convert lists to proper format if needed
        if isinstance(df['durations'].iloc[0], str):
            df['durations'] = df['durations'].apply(eval)
        if isinstance(df['events'].iloc[0], str):
            df['events'] = df['events'].apply(eval)
        
        # Add index column for SAT
        df['index'] = range(len(df))
        
        logger.debug(f"Data columns: {df.columns.tolist()}")
        logger.debug(f"Sample data:\n{df.head()}")
        
        # Save to JSON lines format for SAT
        out_dir = Path(f"{self.processed_dir}/{self.name}")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = Path(f"{out_dir}/{self.name}.json")
        df.to_json(output_file, orient="records", lines=True)
        logger.debug(f"Saved processed data to {output_file}")
        
        # Log statistics
        logger.info(f"Processed {len(df)} patients")
        logger.info(f"Number of events per patient: {len(df['events'].iloc[0])}")
        logger.info(f"Splits: {df['split'].value_counts().to_dict()}")