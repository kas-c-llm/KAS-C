# KAS-C
KAS-C: An LLM-Based Knowledge-Augmented Framework for Assessing Static Analysis Warning Suppressions

## Setup

Install the required Python packages with the same Python version used to run the scripts:

```bash
/usr/local/bin/python3 -m pip install -r requirements.txt
```

## API Keys

Create a `.env` file inside `KAS-C_Implementation/`:

```text
KAS-C_Implementation/.env
```

Add your API keys:

```text
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

Run: 
```bash
/usr/local/bin/python3 KAS-C_Implementation/extract_code.py
```
This reads: Evaluation_Set/KAS_Bench_Post_Cut_Off.csv
and writes: Evaluation_Set/Evaluation_Set.csv which is used as the input for running the classifiers.

Before running classifiers, load the .env file.

Run GPT-5:
```bash
/usr/local/bin/python3 KAS-C_Implementation/GPT5_Classifier.py
```

Run GPT-5 mini:
```bash
/usr/local/bin/python3 KAS-C_Implementation/GPT5mini_suppression_classifier.py
```

Run Claude:
```bash
/usr/local/bin/python3 KAS-C_Implementation/claude_classifier.py
```

