import csv
import os
import re
from pathlib import Path
from typing import TypedDict

# Load .env file (OPENAI_API_KEY)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

from classifier_config import (
    BASE_PROMPT,
    DEFAULT_INPUT_PATH,
    DEFAULT_OUTPUT_GPT5MINI,
    DEFAULT_RAG_PATH,
    SECOND_PROMPT_TEMPLATE_RAG,
)


class ClassificationResult(TypedDict):
    category: str
    reasoning: str
    action: str
    fix_recommendation: str


def _strip_or_empty(v: object) -> str:
    if v is None:
        return ""
    s = str(v)
    return s.strip()


def _read_csv_rows_strip_headers(path: str | Path) -> list[dict]:
    """
    Read a CSV and strip whitespace from header names.
    Avoids issues where the RAG file has headers with extra tabs/spaces.
    """
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader, None)
        if not headers:
            return []
        headers = [h.strip() for h in headers]
        rows: list[dict] = []
        for row in reader:
            if not row:
                continue
            if len(row) < len(headers):
                row = row + [""] * (len(headers) - len(row))
            d = {headers[i]: row[i] for i in range(len(headers))}
            rows.append(d)
        return rows


def _normalize_warning_code(wc: str) -> str:
    return (wc or "").strip()


def _format_rag_context(entry: dict, warning_code: str) -> str:
    """
    Format the KB entry into a compact, model-friendly block.
    NOTE: entry['Category'] here is the *official/tool* category (e.g., STYLE),
    not the output label (False Positive/Unactionable/Technical Debt).
    """
    official_def = _strip_or_empty(entry.get("OfficialDefinition"))
    fp = _strip_or_empty(entry.get("CommonFalsePositive"))
    unactionable = _strip_or_empty(entry.get("CommonUnactionable"))

    parts = [
        f"Warning Code: {warning_code}",
        f"Warning Definition: {official_def or '(missing)'}",
        f"Common False Positive Scenarios: {fp or '(missing)'}",
        f"Common Unactionable Scenarios: {unactionable or '(missing)'}",
    ]
    return "\n".join(parts)


class NewSuppressionClassifier:
    """
    Two-phase classifier:
    - Phase 1: BASE_PROMPT
    - Phase 2: RAG knowledge base entry for the WarningCode
    """

    VALID_CATEGORIES = {"False Positive", "Unactionable", "Technical Debt"}

    OUTPUT_COLUMNS = [
        "a_id",
        "a_commit",
        "a_message",
        "a_file",
        "a_suppressed_annotation",
        "a_line_no",
        "a_diffs",
        "a_post_commit",
        "Category",
        "Category explanation",
        "WarningCode",
        "Code_Snippet",
        "GPT5mini_Output_1",
        "GPT5mini_Output_1_Category",
        "GPT5mini_Output_2",
        "GPT5mini_Output_2_Category",
    ]

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-5-mini",
        rag_path: str | Path | None = None,
    ):
        self.model = model
        self.rag_path = Path(rag_path) if rag_path else self._default_rag_path()
        self._rag_index: dict[str, dict] | None = None
        self._client: OpenAI | None = None

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if key and OpenAI:
            self._client = OpenAI(api_key=key)

    def _default_rag_path(self) -> Path:
        return Path(DEFAULT_RAG_PATH)

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            raise RuntimeError(
                "OpenAI client not initialized. Set OPENAI_API_KEY or pass api_key."
            )
        return self._client

    def _load_rag_index(self) -> dict[str, dict]:
        if self._rag_index is not None:
            return self._rag_index

        if not self.rag_path.exists():
            self._rag_index = {}
            return self._rag_index

        rows = _read_csv_rows_strip_headers(self.rag_path)
        if not rows:
            self._rag_index = {}
            return self._rag_index

        # Accept either "WarningCode" or "Warning Type" as key
        key_col = None
        cols = set(rows[0].keys())
        if "WarningCode" in cols:
            key_col = "WarningCode"
        elif "Warning Type" in cols:
            key_col = "Warning Type"

        if not key_col:
            # Can't index; keep empty
            self._rag_index = {}
            return self._rag_index

        idx: dict[str, dict] = {}
        for r in rows:
            wc = _normalize_warning_code(r.get(key_col, ""))
            if wc:
                idx[wc] = r

        self._rag_index = idx
        return idx

    def _rag_lookup(self, warning_code: str) -> dict | None:
        wc = _normalize_warning_code(warning_code)
        if not wc:
            return None
        idx = self._load_rag_index()
        return idx.get(wc)

    def _parse_response(self, text: str) -> ClassificationResult:
        category = ""
        reasoning = ""
        action = ""
        fix_recommendation = ""

        cat_match = re.search(
            r"Category:\s*(False Positive|Unactionable|Technical Debt)",
            text,
            re.IGNORECASE,
        )
        if cat_match:
            category = cat_match.group(1).strip()
            if category.lower() == "false positive":
                category = "False Positive"
            elif category.lower() == "technical debt":
                category = "Technical Debt"
            elif category.lower() == "unactionable":
                category = "Unactionable"

        reason_match = re.search(r"Reasoning:\s*(.+?)(?=Action:|$)", text, re.DOTALL)
        if reason_match:
            reasoning = reason_match.group(1).strip()

        action_match = re.search(r"Action:\s*(Suppress|Fix)", text, re.IGNORECASE)
        if action_match:
            action = action_match.group(1).strip()

        fix_match = re.search(
            r"Fix recommendation:\s*(.+?)(?=\n\n|\Z)", text, re.DOTALL | re.IGNORECASE
        )
        if fix_match:
            fix_recommendation = fix_match.group(1).strip()

        return ClassificationResult(
            category=category,
            reasoning=reasoning,
            action=action,
            fix_recommendation=fix_recommendation,
        )

    def _call_llm(
        self,
        warning_code: str,
        code_snippet: str,
        file_path: str = "",
    ) -> tuple[str, str]:
        """
        Two-phase LLM chat:
        Phase 1: Base prompt -> output_1
        Phase 2: RAG knowledge base entry for this WarningCode -> output_2

        Returns:
            (output_1, output_2). output_2 is empty when no RAG entry found.
        """
        prompt1 = BASE_PROMPT.format(
            file_path=file_path.strip() or "(not provided)",
            bug_type=warning_code,
            code_snippet=code_snippet.strip() or "(empty snippet)",
        )

        messages = [
            {
                "role": "system",
                "content": "You are an expert Java static analysis classifier. Respond only in the exact format requested.",
            },
            {"role": "user", "content": prompt1},
        ]

        response1 = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        output_1 = response1.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": output_1})

        # Phase 2: RAG
        rag_entry = self._rag_lookup(warning_code)
        if not rag_entry:
            return (output_1, "")

        rag_context = _format_rag_context(rag_entry, warning_code)
        prompt2 = SECOND_PROMPT_TEMPLATE_RAG.format(rag_context=rag_context)
        messages.append({"role": "user", "content": prompt2})

        response2 = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        output_2 = response2.choices[0].message.content or ""
        return (output_1, output_2)

    def classify(
        self,
        warning_code: str,
        code_snippet: str,
        *,
        full_response: bool = False,
        **kwargs,
    ) -> str | ClassificationResult:
        file_path = kwargs.get("a_file", kwargs.get("file_path", ""))
        output_1, output_2 = self._call_llm(warning_code, code_snippet, file_path)
        text = output_2 if output_2 else output_1
        result = self._parse_response(text)
        if full_response:
            return result
        return result["category"]

    def classify_csv(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        limit: int | None = None,
        skip: int = 0,
        existing_output_path: str | Path | None = None,
        filter_warning_codes: list[str] | None = None,
        filter_enabled: bool = False,
    ) -> None:
        with open(input_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)

        if limit is not None:
            rows = all_rows[:limit]
            print(f"Limiting to first {limit} rows")
        else:
            rows = all_rows

        # Normalize filter list
        if filter_warning_codes:
            filter_set = {_normalize_warning_code(wc) for wc in filter_warning_codes if _normalize_warning_code(wc)}
        else:
            filter_set = set()

        output_rows: list[dict] = []

        if skip > 0 and existing_output_path and Path(existing_output_path).exists():
            with open(existing_output_path, "r", encoding="utf-8-sig", newline="") as f:
                existing_reader = csv.DictReader(f)
                existing_list = list(existing_reader)
            output_rows = existing_list[:skip]
            rows_to_process = rows[skip:]
            print(
                f"Skipping first {skip} rows (using existing output), processing {len(rows_to_process)} rows"
            )
        else:
            rows_to_process = rows
            if skip > 0:
                print(
                    f"Skip={skip} but no existing output found; processing all {len(rows_to_process)} rows"
                )

        def _write_output(rows_to_write: list[dict], path: str | Path) -> None:
            with open(path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=self.OUTPUT_COLUMNS, extrasaction="ignore"
                )
                writer.writeheader()
                writer.writerows(rows_to_write)

        for i, row in enumerate(rows_to_process):
            warning_code = row.get("WarningCode", "")
            code_snippet = row.get("Code_Snippet", "") or ""
            file_path = row.get("a_file", "") or ""
            row_num = skip + i + 1

            # Optional filtering by WarningCode
            if filter_enabled and filter_set:
                if _normalize_warning_code(warning_code) not in filter_set:
                    # Keep row but leave GPT outputs empty so evaluation can see blanks
                    output_row = {
                        "a_id": row.get("a_id", ""),
                        "a_commit": row.get("a_commit", ""),
                        "a_message": row.get("a_message", ""),
                        "a_file": row.get("a_file", ""),
                        "a_suppressed_annotation": row.get("a_suppressed_annotation", ""),
                        "a_line_no": row.get("a_line_no", ""),
                        "a_diffs": row.get("a_diffs", ""),
                        "a_post_commit": row.get("a_post_commit", ""),
                        "Category": row.get("Category", ""),
                        "Category explanation": row.get("Category explanation", ""),
                        "WarningCode": warning_code,
                        "Code_Snippet": code_snippet,
                        "GPT5mini_Output_1": "",
                        "GPT5mini_Output_1_Category": "",
                        "GPT5mini_Output_2": "",
                        "GPT5mini_Output_2_Category": "",
                    }
                    output_rows.append(output_row)
                    print(f"  Skipped row {row_num}/{len(rows)} (WarningCode filter)")
                    continue

            try:
                output_1, output_2 = self._call_llm(
                    warning_code, code_snippet, file_path
                )
            except Exception as e:
                _write_output(output_rows, output_path)
                print(f"\nAPI error at row {row_num}/{len(rows)}: {e}")
                print(
                    f"Saved {len(output_rows)} rows to {output_path}. Resume later with SKIP={len(output_rows)}."
                )
                raise

            parsed_1 = self._parse_response(output_1)
            parsed_2 = self._parse_response(output_2)

            output_row = {
                "a_id": row.get("a_id", ""),
                "a_commit": row.get("a_commit", ""),
                "a_message": row.get("a_message", ""),
                "a_file": row.get("a_file", ""),
                "a_suppressed_annotation": row.get("a_suppressed_annotation", ""),
                "a_line_no": row.get("a_line_no", ""),
                "a_diffs": row.get("a_diffs", ""),
                "a_post_commit": row.get("a_post_commit", ""),
                "Category": row.get("Category", ""),
                "Category explanation": row.get("Category explanation", ""),
                "WarningCode": warning_code,
                "Code_Snippet": code_snippet,
                "GPT5mini_Output_1": output_1,
                "GPT5mini_Output_1_Category": parsed_1.get("category", ""),
                "GPT5mini_Output_2": output_2,
                "GPT5mini_Output_2_Category": parsed_2.get("category", ""),
            }
            output_rows.append(output_row)
            print(f"  Processed row {row_num}/{len(rows)}")

        _write_output(output_rows, output_path)
        print(f"Wrote {len(output_rows)} rows to {output_path}")


if __name__ == "__main__":
    import sys

    # Usage from terminal or PyCharm:
    #   python GPT5mini_suppression_classifier.py
    # Requires:
    #   OPENAI_API_KEY in environment or .env

    rag_override: str | None = None
    filter_codes_arg: str | None = None
    filter_enabled = False

    args = list(sys.argv[1:])

    # Parse --rag
    if "--rag" in args:
        idx = args.index("--rag")
        if idx + 1 < len(args):
            rag_override = args[idx + 1]
            del args[idx : idx + 2]

    # Parse --filter-warning-codes CODE1,CODE2,...
    if "--filter-warning-codes" in args:
        idx = args.index("--filter-warning-codes")
        if idx + 1 < len(args):
            filter_codes_arg = args[idx + 1]
            del args[idx : idx + 2]
            filter_enabled = True  # enabling filter when codes are provided

    # Optional explicit --filter-on / --no-filter
    if "--filter-on" in args:
        filter_enabled = True
        args.remove("--filter-on")
    if "--no-filter" in args:
        filter_enabled = False
        args.remove("--no-filter")

    default_input = DEFAULT_INPUT_PATH
    default_output = DEFAULT_OUTPUT_GPT5MINI

    input_file = args[0] if len(args) >= 1 else default_input
    output_file = args[1] if len(args) >= 2 else default_output

    # Filter only when --filter-warning-codes is explicitly passed; otherwise process all rows
    if filter_codes_arg:
        parts = [p.strip() for p in filter_codes_arg.split(",")]
        filter_codes: list[str] | None = [p for p in parts if p]
    else:
        filter_codes = None

    classifier = NewSuppressionClassifier(rag_path=rag_override)

    if classifier._client is None:
        print("Set OPENAI_API_KEY to run classification.")
        print("Example: export OPENAI_API_KEY='your-key'")
        raise SystemExit(1)

    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"RAG: {classifier.rag_path}")
    print("Processing...")
    if filter_enabled and filter_codes:
        print(f"WarningCode filter ON for: {', '.join(filter_codes)}")
    else:
        print("WarningCode filter OFF (processing all rows)")

    classifier.classify_csv(
        input_file,
        output_file,
        filter_warning_codes=filter_codes,
        filter_enabled=filter_enabled,
    )

