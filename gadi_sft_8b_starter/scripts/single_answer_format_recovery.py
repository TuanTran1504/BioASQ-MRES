"""Conservative, gold-blind recovery of a single Answer: line for diagnostics."""
import re

from run_single_answer_sampling import parse_single_answer

VERSION = "single-answer-line-format-recovery-v1"


def recover_single_answer(raw):
    try:
        return parse_single_answer(raw), "strict_tagged"
    except ValueError:
        pass
    match = re.fullmatch(r"Answer:\s*(.*)", raw.strip(), flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None, "unrecoverable"
    answer = match[1].strip()
    if answer.startswith("[BE]"):
        answer = answer[4:].strip()
    if answer.endswith("[EE]"):
        answer = answer[:-4].strip()
    if (not answer or "\n" in answer or "\r" in answer or "[BE]" in answer or "[EE]" in answer
            or re.search(r"\bAnswer:", answer, flags=re.IGNORECASE)):
        return None, "unrecoverable"
    # Preserve the entire expression, including qualifiers and parenthetical
    # names. Never split synonyms, trim explanatory sentences, or consult gold.
    return re.sub(r"\s+", " ", answer), "single_answer_line"
