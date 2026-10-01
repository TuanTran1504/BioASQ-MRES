from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utility.eval_runner import run_evaluation as _run_evaluation


SUPPORTED_QUESTION_TYPES = ("summary", "factoid", "list", "yesno")


def flatten_arg_groups(values: list[list[str]] | None) -> list[str] | None:
    if values is None:
        return None
    flattened = [item for group in values for item in group]
    return flattened or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one or more BioASQ answer-generation models against raw "
            "Task 13B JSON files. Model references can come from the current "
            "model registry (run ids or aliases) or from direct model paths."
        )
    )
    parser.add_argument(
        "--eval-input",
        nargs="+",
        default=None,
        help=(
            "One or more evaluation files. Supports raw BioASQ JSON with a "
            "'questions' list or prepared JSON lists. Defaults to all JSON files "
            "under data/Task13BTest."
        ),
    )
    parser.add_argument(
        "--model-ref",
        action="append",
        nargs="+",
        default=None,
        help=(
            "Model references to evaluate. Each value may be a registry alias, a "
            "registry run id, a local path, a Hugging Face model name, or an "
            "OpenAI model written as openai:<model-name>. This "
            "flag can be passed once with multiple values or repeated."
        ),
    )
    parser.add_argument(
        "--openai-model",
        action="append",
        default=None,
        help=(
            "OpenAI model to evaluate as a candidate through the same prompt, "
            "official scorer, and optional semantic judge. Repeat for multiple models."
        ),
    )
    parser.add_argument(
        "--api-key-file",
        default="open_ai_api.txt",
        help="File containing the API key used for OpenAI candidate generation.",
    )
    parser.add_argument(
        "--openai-endpoint",
        default="https://api.openai.com/v1/chat/completions",
        help="Chat-completions endpoint used for OpenAI candidate generation.",
    )
    parser.add_argument(
        "--max-new-api-calls",
        type=int,
        default=0,
        help=(
            "Hard shared HTTP-call budget for all OpenAI candidate models. "
            "Required when --openai-model is used; cached responses do not count."
        ),
    )
    parser.add_argument("--api-timeout-seconds", type=int, default=180)
    parser.add_argument("--api-request-delay-seconds", type=float, default=0.0)
    parser.add_argument("--api-max-retries", type=int, default=2)
    parser.add_argument(
        "--openai-cache-dir",
        default=None,
        help=(
            "Optional prior OpenAI candidate cache directory to reuse. New responses "
            "are written inside the current evaluation run."
        ),
    )
    parser.add_argument(
        "--all-registry-runs",
        action="store_true",
        help="Evaluate all completed answer_generation runs from the model registry.",
    )
    parser.add_argument(
        "--question-types",
        nargs="+",
        default=list(SUPPORTED_QUESTION_TYPES),
        choices=list(SUPPORTED_QUESTION_TYPES),
        help="Question types to evaluate.",
    )
    parser.add_argument(
        "--max-resources",
        type=int,
        default=3,
        help="Maximum number of PubMed resources to include per example. Use 0 for all resources.",
    )
    parser.add_argument(
        "--max-resource-chars",
        type=int,
        default=1200,
        help="Maximum characters per serialized PubMed resource. Use 0 for no truncation.",
    )
    parser.add_argument(
        "--resource-selection",
        default="first",
        choices=["first", "embedding"],
        help=(
            "How to choose which resources to include. 'first' keeps the current "
            "dataset order; 'embedding' reranks by question-resource similarity "
            "before applying --max-resources."
        ),
    )
    parser.add_argument(
        "--resource-granularity",
        default="document",
        choices=["document", "snippet"],
        help=(
            "Whether evaluation resources are grouped by document or treated as "
            "individual snippets before selection."
        ),
    )
    parser.add_argument(
        "--resource-window-mode",
        default="single",
        choices=["single", "sequential"],
        help=(
            "How to feed selected resources to the model. 'single' keeps the "
            "existing behavior and builds one prompt with up to --max-resources "
            "items. 'sequential' first collects the full selected resource list, "
            "then evaluates the model over sequential windows of size "
            "--max-resources and aggregates the window-level answers with "
            "--aggregation-strategy."
        ),
    )
    parser.add_argument(
        "--resource-window-step",
        type=int,
        default=0,
        help=(
            "Step size between sequential resource windows. Use 0 to default to "
            "--max-resources, which yields non-overlapping windows."
        ),
    )
    parser.add_argument(
        "--resource-reranker-model",
        default="sentence-transformers/all-MiniLM-L12-v2",
        help=(
            "Embedding query model or single-encoder model used when "
            "--resource-selection embedding is enabled."
        ),
    )
    parser.add_argument(
        "--resource-reranker-article-model",
        default=None,
        help=(
            "Optional separate embedding model used for resources/documents when "
            "--resource-selection embedding is enabled. Use this for dual-encoder "
            "retrievers such as MedCPT."
        ),
    )
    parser.add_argument(
        "--resource-reranker-device",
        default="auto",
        help=(
            "Device used for the resource reranker, for example 'cpu', 'cuda', or "
            "'auto'. The default 'auto' uses CUDA when available and otherwise falls "
            "back to CPU."
        ),
    )
    parser.add_argument(
        "--resource-reranker-batch-size",
        type=int,
        default=32,
        help="Batch size used when encoding candidate resources for reranking.",
    )
    parser.add_argument(
        "--max-summary-answers",
        type=int,
        default=1,
        help="Compatibility option used when converting raw BioASQ rows.",
    )
    parser.add_argument(
        "--max-factoid-answers",
        type=int,
        default=5,
        help="Maximum number of exact-answer variants expected in factoid outputs.",
    )
    parser.add_argument(
        "--max-list-items",
        type=int,
        default=100,
        help="Maximum number of exact-answer items expected in list outputs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of evaluation examples.",
    )
    parser.add_argument(
        "--registry-path",
        default="models/registry.json",
        help="Repo-relative model registry JSON path.",
    )
    parser.add_argument(
        "--prompt-registry-path",
        default=None,
        help=(
            "Optional repo-relative prompt registry JSON path. If omitted, the "
            "script falls back to the built-in QUESTION_INSTRUCTIONS."
        ),
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help=(
            "Optional repo-relative JSON file describing one prompt bundle. This "
            "can be either a single inline prompt object or a full prompt registry."
        ),
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Optional prompt alias or prompt_id to resolve from the prompt registry.",
    )
    parser.add_argument(
        "--summary-reference-mode",
        default="first",
        choices=["first", "all-max", "all-mean"],
        help=(
            "How to use multiple ideal-answer references when computing summary metrics. "
            "'first' matches the repo's existing data-prep assumption; 'all-max' and "
            "'all-mean' compare against all available references."
        ),
    )
    parser.add_argument(
        "--chat-template",
        default=None,
        help="Optional chat template override used when the model does not provide one.",
    )
    parser.add_argument(
        "--prompt-format",
        default=None,
        choices=["chat", "unitor_plain"],
        help="Optional prompt format override used when the model manifest does not provide one.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
        help="Max sequence length passed to the model loader when supported.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate per question.",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=1,
        help=(
            "Number of generations to sample per question. Values above 1 aggregate "
            "all generated answers into one final prediction before scoring."
        ),
    )
    parser.add_argument(
        "--aggregation-strategy",
        default="union",
        choices=["union", "frequency"],
        help="How to combine multiple generations into one final answer.",
    )
    parser.add_argument(
        "--aggregation-min-frequency",
        type=int,
        default=2,
        help=(
            "Minimum number of samples an item must appear in when "
            "--aggregation-strategy frequency is used."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Set > 0 and use --do-sample to sample.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling parameter used when --do-sample is set.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling during generation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed used for stochastic decoding runs.",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        choices=["float16", "bfloat16", "float32"],
        help="Optional dtype override for loaders that support it.",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit loading for Unsloth-based loaders.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map passed to Transformers/PEFT fallback loaders.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Force model/tokenizer loading from local files and Hugging Face cache only.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional directory where all evaluation artifacts will be written. "
            "Defaults to Artifacts/evaluations/<timestamp>-<prompt-id>."
        ),
    )
    parser.add_argument(
        "--score-backend",
        default="bioasq_java",
        choices=["none", "bioasq_java"],
        help=(
            "Use the bundled official BioASQ Phase B Java evaluator, or 'none' "
            "to save generations without calculating metrics."
        ),
    )
    parser.add_argument(
        "--semantic-judge",
        action="store_true",
        help=(
            "Add snippet-grounded semantic accuracy/MRR. Exact aliases are handled "
            "deterministically; non-exact factoid answers are classified by an LLM."
        ),
    )
    parser.add_argument(
        "--semantic-judge-model",
        default="gpt-4.1-mini-2025-04-14",
        help="OpenAI model used only as the snippet-grounded evaluator.",
    )
    parser.add_argument(
        "--semantic-judge-api-key-file",
        default="open_ai_api.txt",
        help="File containing the API key used by the semantic judge.",
    )
    parser.add_argument(
        "--semantic-judge-endpoint",
        default="https://api.openai.com/v1/chat/completions",
    )
    parser.add_argument(
        "--semantic-judge-max-new-calls",
        type=int,
        default=0,
        help=(
            "Hard shared HTTP-call budget across all evaluated candidate models. "
            "Exact matches and cached judgments do not count."
        ),
    )
    parser.add_argument("--semantic-judge-timeout-seconds", type=int, default=180)
    parser.add_argument("--semantic-judge-request-delay-seconds", type=float, default=0.0)
    parser.add_argument("--semantic-judge-max-retries", type=int, default=2)
    parser.add_argument(
        "--semantic-judge-cache-dir",
        default=None,
        help=(
            "Optional prior semantic-judge cache directory to reuse. New responses "
            "are always written inside the current evaluation run."
        ),
    )
    parser.add_argument(
        "--bioasq-java-jar",
        default="third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar",
        help="Repo-relative path to the official BioASQ Java evaluator JAR.",
    )
    parser.add_argument(
        "--bioasq-java-version",
        type=int,
        default=5,
        choices=[2, 3, 5, 8, 9],
        help=(
            "Challenge format version passed to EvaluatorTask1b -e. "
            "The current repo's BioASQ task-b exact-answer format should use 5."
        ),
    )
    parser.add_argument(
        "--bioasq-java-heap",
        default="4G",
        help="Java heap size passed to the official evaluator, for example 4G or 8G.",
    )
    args = parser.parse_args()
    args.model_ref = flatten_arg_groups(args.model_ref)
    local_model_refs: list[str] = []
    openai_model_refs: list[str] = list(args.openai_model or [])
    for model_ref in args.model_ref or []:
        if str(model_ref).casefold().startswith("openai:"):
            model_name = str(model_ref).split(":", 1)[1].strip()
            if not model_name:
                parser.error("OpenAI model references must use openai:<model-name>.")
            openai_model_refs.append(model_name)
        else:
            local_model_refs.append(str(model_ref))
    args.model_ref = local_model_refs or None
    args.openai_model = list(dict.fromkeys(openai_model_refs)) or None
    if not args.model_ref and not args.all_registry_runs and not args.openai_model:
        parser.error("Select at least one --model-ref, --openai-model, or --all-registry-runs.")
    if args.openai_model and args.max_new_api_calls <= 0:
        parser.error("--openai-model requires --max-new-api-calls > 0.")
    if args.semantic_judge and args.semantic_judge_max_new_calls <= 0:
        parser.error("--semantic-judge requires --semantic-judge-max-new-calls > 0.")
    if args.resource_window_mode != "single":
        if int(args.max_resources or 0) <= 0:
            parser.error("--resource-window-mode sequential requires --max-resources > 0.")
        if int(args.resource_window_step or 0) < 0:
            parser.error("--resource-window-step must be >= 0.")
        if int(args.resource_window_step or 0) > int(args.max_resources or 0):
            parser.error(
                "--resource-window-step cannot be greater than --max-resources, "
                "otherwise some resources would be skipped."
            )
    return args


def run_evaluation(args: argparse.Namespace) -> None:
    _run_evaluation(args, supported_question_types=SUPPORTED_QUESTION_TYPES)
