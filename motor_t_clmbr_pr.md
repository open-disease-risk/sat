# Motor-T and CLMBR Integration with Embedding Database Support

## Summary

This PR integrates Motor-T and CLMBR models into the SAT framework, adding support for both transformer-based models and pre-computed embedding databases for multi-event survival analysis.

### Key Features Added

- **Motor-T Integration**: Support for Stanford's medical sequence transformer model
- **CLMBR Embedding Support**: Integration with pre-computed 768-dimensional CLMBR embeddings
- **Multi-Model Architecture**: Unified framework supporting both trainable transformers and frozen embeddings
- **EHRSHOT Dataset Support**: Complete pipeline for the 14,251 patient multi-event survival dataset
- **Robust Evaluation Pipeline**: Fixed patient_id and anchor_time passing for embedding models

### Major Components

#### Model Support
- `sat/models/femr_wrapper.py` - FEMR model detection and wrapping
- `sat/models/embedding_wrapper.py` - Database-backed embedding lookup
- Enhanced MTL head with smart freezing logic for different model types

#### Data Processing
- `scripts/prepare_ehrshot_data.py` - EHRSHOT sequence extraction with proper event mappings
- `sat/data/dataset/ehrshot_cohort.py` - Multi-event cohort creation with true survival times
- `sat/data/dataset/parse_ehrshot_motor.py` - MOTOR-T specific data formatting

#### Configurations
- `conf/experiments/ehrshot_motor/` - Motor-T experiment configs
- `conf/experiments/ehrshot_clmbr/` - CLMBR embedding experiment configs
- `conf/transformers/motor_t.yaml` - Motor-T model configuration
- `conf/transformers/embedding_db.yaml` - Embedding database configuration

#### Evaluation Fixes
- Fixed import paths (`sat.models.heads` instead of `sat.models.tasks.heads`)
- Enhanced survival analysis pipeline to handle patient_id and anchor_time
- Custom `FullRecordDatasetColumn` for embedding database evaluation
- Improved batch tokenization handling

#### Utility Scripts
- `scripts/convert_motor_t_to_pytorch.py` - JAX to PyTorch model conversion
- `scripts/fix_tokenizer_special_tokens.py` - Tokenizer repair utility with CLI args
- `scripts/test_embedding_pipeline.py` - End-to-end embedding pipeline verification

### Technical Improvements

- **Smart Model Freezing**: Automatically detects embedding models and skips freezing (no trainable parameters)
- **Custom Data Collation**: `EmbeddingDataCollator` preserves patient_id and anchor_time during training
- **Pipeline Registration**: Proper HuggingFace pipeline integration for survival analysis
- **Robust Error Handling**: Better handling of missing files and configuration issues

### Commit History

1. **fix: improve MTL freezing logic for embedding models** (`da93d174`)
   - Skip freezing for embedding database models (no trainable parameters)
   - Use base_model attribute when available for freezing
   - Add custom EmbeddingDataCollator for patient_id and anchor_time handling

2. **fix: update CLMBR configs for embedding database** (`260515ad`)
   - Set freeze_transformer to false (no trainable parameters to freeze)
   - Remove redundant task config overrides
   - Use ehrshot_motor data/tokenizers for consistency

3. **fix: resolve evaluation pipeline issues for embedding models** (`a95a107b`)
   - Fix import paths in eval.py and infer.py (sat.models.heads)
   - Add pipeline registration import for survival-analysis
   - Improve batch tokenization handling for single examples
   - Enhance survival analysis pipeline to extract patient_id and anchor_time
   - Create FullRecordDatasetColumn for embedding database evaluation

4. **enhance: improve utility scripts and data processing** (`d3125d35`)
   - Add command-line arguments to fix_tokenizer_special_tokens.py
   - Enhance EHRSHOT data preparation script with better error handling
   - Add comprehensive embedding pipeline test script
   - Improve Motor-T to PyTorch conversion script

5. **refactor: clean up data processing and loss code** (`8fc7acb9`)
   - Improve EHRSHOT cohort data processing formatting
   - Fix import ordering in parse_ehrshot_motor.py
   - Clean up loss function code style

6. **fix: complete evaluator import fixes** (`15b78072`)
   - Update remaining import in survival_analysis evaluator
   - Complete the evaluation pipeline fixes

### Testing Status

- ✅ CLMBR evaluation pipeline working with patient_id/anchor_time passing
- ✅ Motor-T configuration and model loading verified
- ✅ Multi-event survival analysis with 7 outcomes (hypertension, hyperlipidemia, acute MI, pancreatic cancer, lupus, celiac disease, length of stay)
- ✅ Code quality checks (isort, ruff) passing
- ✅ Pipeline registration and HuggingFace integration working

### Key Issue Resolved

The major breakthrough was fixing the evaluation pipeline for embedding database models. Previously, the pipeline only passed text data, but embedding models require `patient_id` and `anchor_time` to lookup pre-computed embeddings. This was resolved by:

1. Creating a custom `FullRecordDatasetColumn` that passes complete records instead of just text
2. Enhancing the survival analysis pipeline to extract `patient_id` and `anchor_time` from input dictionaries
3. Converting `patient_id` to proper tensor format for model consumption

### Files Changed
- 44 files modified/added
- 3,489 insertions, 85 deletions
- Major additions: embedding support, EHRSHOT integration, evaluation fixes

### Usage Examples

#### Running Motor-T Experiments
```bash
# Training
poetry run python -m sat.finetune experiments=ehrshot_motor/survival

# Evaluation
poetry run python -m sat.eval experiments=ehrshot_motor/survival
```

#### Running CLMBR Experiments
```bash
# Training (with embedding database)
poetry run python -m sat.finetune experiments=ehrshot_clmbr/survival

# Evaluation
poetry run python -m sat.eval experiments=ehrshot_clmbr/survival
```

#### Testing Embedding Pipeline
```bash
poetry run python scripts/test_embedding_pipeline.py
```

### Dependencies
- Requires EHRSHOT dataset and CLMBR embeddings to be available
- Motor-T model conversion from JAX weights using provided script
- Updated tokenizer configurations for medical code sequences

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>