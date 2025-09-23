# Setting up MOTOR-T Access

## Step 1: Request Access on HuggingFace
1. Go to https://huggingface.co/StanfordShahLab/motor-t-base
2. Click "Request access" 
3. Fill out the form and wait for approval

## Step 2: Get your HuggingFace Token
1. Go to https://huggingface.co/settings/tokens
2. Create a new token with "read" permissions
3. Copy the token

## Step 3: Login to HuggingFace CLI
```bash
# Install huggingface-hub if needed
pip install -U huggingface-hub

# Login with your token
huggingface-cli login
# Paste your token when prompted
```

## Step 4: Alternative - Use Token in Code
If you can't use the CLI, set an environment variable:
```bash
export HF_TOKEN=your_token_here
```

Or use it directly in the config by updating `conf/transformers/motor_t.yaml`:
```yaml
pretrained_model_name_or_path: "StanfordShahLab/motor-t-base"
trust_remote_code: true
use_auth_token: true  # Will use HF_TOKEN env var
```

## Step 5: Verify Access
Once you have access and are logged in:
```bash
poetry run python -c "
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained('StanfordShahLab/motor-t-base', trust_remote_code=True)
print('✓ MOTOR-T loaded successfully!')
"
```

## Current Workaround
Until you have MOTOR-T access, the config uses BERT-base-uncased as a placeholder model with similar architecture.