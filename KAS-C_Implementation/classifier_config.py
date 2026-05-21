"""
Shared configuration for suppression classifiers (OpenAI and Claude).

Edit this file to change prompts, default input, RAG path, or output paths
without modifying the individual classifier scripts.
"""

# --- File paths ---
DEFAULT_INPUT_PATH = "Evaluation_Set/Evaluation_Set.csv"
DEFAULT_RAG_PATH = "Knowledge_Base.csv"
# GPT-5 (non-mini) `GPT5_Classifier.py` → columns GPT5_Output_*
DEFAULT_OUTPUT_OPENAI = "Evaluation_Set/GPT5_Output.csv"
# GPT-5 mini `GPT5mini_suppression_classifier.py` → columns GPT5mini_Output_*
DEFAULT_OUTPUT_GPT5MINI = "Evaluation_Set/GPT5mini_Output.csv"
DEFAULT_OUTPUT_CLAUDE = "Evaluation_Set/Claude_Output.csv"


# --- Prompts ---
BASE_PROMPT = """You are an expert in Java code analysis and static analyzer warnings. Categorize the warning using ONLY the provided code snippet, associated context and bug type.

CATEGORIES:
1. False Positive - The analyzer incorrectly reports the warning due to its limitations; the code is actually safe.
2. Unactionable - The warning is correctly identified, but the developer intentionally accepts it due to design decisions, external validation (outside code analysis scope), or because it appears in test, auto-generated, or third-party code.
3. Technical Debt - The warning is correctly identified and should be fixed.

INPUT:
- File path: {file_path}
- Bug Type: {bug_type}
- Code Snippet:
{code_snippet}

OUTPUT FORMAT (follow exactly; no extra texts; 1–2 lines per point; Include a confidence score based on your analysis):
Category: [False Positive | Unactionable | Technical Debt]
Reasoning: (1) Explain the issue the static analyzer detected and why this code was flagged.
(2) Justify why the selected category fits (mention what in the snippet supports it).
(3) Action: [Suppress | Fix]
(4) Fix recommendation: [specific code change if Fix; otherwise write "N/A"]
(5) Confidence: [0–100]% """


SECOND_PROMPT_TEMPLATE_RAG = """You are given reference information about a static analysis warning type. This includes:
- Definition of the warning,
- Common False Positive patterns,
- Common Unactionable patterns.

Use this reference to understand the intent and reasoning behind each category.

INSTRUCTIONS:
1. First, analyze the code: identify what it does and why the static analyzer flagged it.

2. Then check whether the warning matches a False Positive or Unactionable situation:
   - Focus on contextual semantics, not exact code patterns.
   - Do not match specific examples directly. Instead, determine whether the situation is conceptually similar.
   - Use the warning definition to understand what is being checked.
   - There may be other valid False Positive or Unactionable scenarios not explicitly listed.

3. If the warning is not convincingly False Positive or Unactionable, classify it as Technical Debt.

4. Classification rules:
   - False Positive - The analyzer incorrectly reports the warning due to its limitations; the code is actually safe.
   - Unactionable - The warning is correctly identified, but the developer intentionally accepts it due to design decisions, external validation (outside code analysis scope), or because it appears in test, auto-generated, or third-party code.
   - Technical Debt - The warning is correctly identified and should be fixed.

5. If unsure, prioritize the actual code over the examples.

REFERENCE INFORMATION:
{rag_context}

Now provide your FINAL answer in the exact format specified earlier including the new Confidence score of your analysis."""
