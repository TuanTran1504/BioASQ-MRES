from __future__ import annotations

import argparse


TASK_NAME = "answer_generation"


QUESTION_INSTRUCTIONS = {
    "summary": (
        "You are a biomedical expert. Your task is to generate a concise, "
        "well-structured summary answering the given question. Base your response "
        "on the provided PubMed resources, focusing on the text marked with [BS] "
        "and [ES].\n"
        "Rules:\n"
        "- Use only the information provided with a special focus on the marked information.\n"
        "- The summary must be at most 200 words.\n"
        "- Do NOT include personal opinions, speculations, or unrelated information.\n"
        "- Maintain a neutral and scientific tone."
    ),
    "factoid": (
        "You are a biomedical expert. Your task is to extract the most relevant "
        "factoid-based answer from the provided PubMed resources. Relevant "
        "information is marked with [BS] and [ES].\n"
        "Rules:\n"
        "- Use only the information provided with a special focus on the marked information.\n"
        "- Return up to 5 short expressions.\n"
        "- You may return fewer than 5 expressions when fewer are well-supported by the evidence.\n"
        "- Do NOT invent filler answers just to reach 5 outputs.\n"
        "- If the answer has more than one expression, order them by decreasing confidence.\n"
        "- Often the expressions represent the same concept but written differently.\n"
        "- Do NOT provide explanations or extra text.\n"
        "- Maintain this format strictly: [BE] short answer [EE] [BE] alternate wording [EE] ..."
    ),
    "list": (
        "You are a biomedical expert. Your task is to generate a complete list of "
        "relevant items, jointly taken to constitute a single answer, based on the "
        "provided PubMed resources. Relevant information is marked with [BS] and [ES].\n"
        "Rules:\n"
        "- Use only the information provided with a special focus on the marked information.\n"
        "- Do NOT provide explanations or extra text.\n"
        "- The returned list must contain no more than 100 entries of no more than 100 characters each.\n"
        "- Maintain this format strictly: [BI] list item [EI] [BI] another list item [EI] ..."
    ),
    "yesno": (
        "You are a biomedical expert. Your task is to answer a yes/no question "
        "based on the provided PubMed resources. Relevant information is marked "
        "with [BS] and [ES].\n"
        "Rules:\n"
        "- Use only the information provided with a special focus on the marked information.\n"
        "- Do NOT provide explanations or extra text.\n"
        "- Answer only 'yes' or 'no'."
    ),
}

PROMPT_FORMAT_CHAT = "chat"
PROMPT_FORMAT_UNITOR_PLAIN = "unitor_plain"

TRAINING_PRESET_DEFAULT = "default"
TRAINING_PRESET_UNITOR_LLAMA31 = "unitor-llama31-answer-gen"


def training_preset_defaults(preset: str) -> dict[str, object]:
    if preset != TRAINING_PRESET_UNITOR_LLAMA31:
        return {}

    return {
        "model_name": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "question_types": ["summary", "factoid", "list", "yesno"],
        "prompt_format": PROMPT_FORMAT_UNITOR_PLAIN,
        "chat_template": "llama-3",
        "max_resources": 5,
        "max_seq_length": 1024,
        "lora_r": 32,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 5,
        "num_train_epochs": 2.0,
        "learning_rate": 6e-4,
        "weight_decay": 0.01,
        "dataset_num_proc": 1,
        "response_template": "\n# Answer:",
        "response_template_trim_tokens": 2,
    }


def parse_args() -> argparse.Namespace:
    preset_parser = argparse.ArgumentParser(add_help=False)
    preset_parser.add_argument(
        "--preset",
        default=TRAINING_PRESET_DEFAULT,
        choices=[TRAINING_PRESET_DEFAULT, TRAINING_PRESET_UNITOR_LLAMA31],
    )
    preset_args, _ = preset_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description=(
            "Supervised fine-tuning for BioASQ answer generation with Unsloth "
            "models. Accepts either raw BioASQ JSON files or pre-built JSON "
            "records in the style of the UNITOR repository."
        )
    )
    parser.add_argument(
        "--preset",
        default=TRAINING_PRESET_DEFAULT,
        choices=[TRAINING_PRESET_DEFAULT, TRAINING_PRESET_UNITOR_LLAMA31],
        help=(
            "Optional training preset. "
            f"'{TRAINING_PRESET_UNITOR_LLAMA31}' matches the released UNITOR "
            "LLaMA answer-generation script."
        ),
    )
    parser.add_argument(
        "--train-input",
        nargs="+",
        required=True,
        help="One or more training files. Supports raw BioASQ JSON or prepared JSON lists.",
    )
    parser.add_argument(
        "--eval-input",
        nargs="+",
        default=None,
        help="Optional evaluation files. If omitted, a split is created from train-input.",
    )
    parser.add_argument(
        "--question-types",
        nargs="+",
        default=["factoid", "list", "yesno"],
        choices=sorted(QUESTION_INSTRUCTIONS.keys()),
        help="Question types to keep when preparing data from raw BioASQ files.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
        help="Holdout ratio used when eval-input is not provided.",
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
        "--max-summary-answers",
        type=int,
        default=1,
        help="How many ideal answers to keep for summary questions from raw BioASQ files.",
    )
    parser.add_argument(
        "--max-factoid-answers",
        type=int,
        default=5,
        help="Maximum number of exact-answer variants to keep for factoid questions.",
    )
    parser.add_argument(
        "--max-list-items",
        type=int,
        default=100,
        help="Maximum number of exact-answer items to keep for list questions.",
    )
    parser.add_argument(
        "--prepared-output-dir",
        default=None,
        help="Optional directory where the prepared train/eval JSON files will be saved.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap for training examples, useful for smoke tests.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Optional cap for evaluation examples, useful for smoke tests.",
    )
    parser.add_argument(
        "--model-name",
        default="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        help="Base model to load from Unsloth.",
    )
    parser.add_argument(
        "--chat-template",
        default="llama-3",
        help="Chat template name passed to Unsloth.",
    )
    parser.add_argument(
        "--prompt-format",
        default=PROMPT_FORMAT_CHAT,
        choices=[PROMPT_FORMAT_CHAT, PROMPT_FORMAT_UNITOR_PLAIN],
        help="Prompt/rendering format used for training and later evaluation.",
    )
    parser.add_argument(
        "--prompt-registry-path",
        default=None,
        help=(
            "Optional repo-relative prompt registry JSON path. If omitted, raw "
            "BioASQ data prep falls back to the built-in QUESTION_INSTRUCTIONS."
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
        "--instruction-part",
        default=None,
        help="Optional override for train_on_responses_only instruction marker.",
    )
    parser.add_argument(
        "--response-part",
        default=None,
        help="Optional override for train_on_responses_only response marker.",
    )
    parser.add_argument(
        "--response-template",
        default="\nAnswer:",
        help=(
            "Template used by completion-only masking. "
            "UNITOR uses '\\n# Answer:'."
        ),
    )
    parser.add_argument(
        "--response-template-trim-tokens",
        type=int,
        default=0,
        help=(
            "How many leading tokens to trim from the encoded response template "
            "before passing it to the completion-only collator."
        ),
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=1024,
        help="Maximum sequence length.",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        choices=["float16", "bfloat16"],
        help="Optional dtype override.",
    )
    parser.add_argument(
        "--save-dtype",
        default="float32",
        choices=["auto", "float16", "bfloat16", "float32", "fp16", "bf16", "fp32"],
        help=(
            "Dtype used when saving the final adapter. Defaults to float32 for "
            "stable, portable LoRA artifacts. Use 'auto' to preserve the current "
            "in-memory dtype."
        ),
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit loading.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only load the base model/tokenizer from the local Hugging Face cache.",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha.",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.0,
        help="LoRA dropout.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=2,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=2,
        help="Per-device eval batch size.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="Warmup steps.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=2.0,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=1,
        help="Logging frequency.",
    )
    parser.add_argument(
        "--save-strategy",
        default="epoch",
        choices=["epoch", "steps"],
        help=(
            "How often to save SFT checkpoints. Generated-dev checkpoint selection "
            "also runs on this cadence."
        ),
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=50,
        help="Checkpoint interval when --save-strategy steps is used.",
    )
    parser.add_argument(
        "--eval-strategy",
        default="auto",
        choices=["auto", "epoch", "steps", "no"],
        help=(
            "How often to run trainer-side eval_loss validation. 'auto' matches the "
            "save strategy whenever an eval set is available so validation loss is "
            "logged alongside generated dev metrics."
        ),
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=None,
        help=(
            "Eval interval when --eval-strategy steps is used. Defaults to "
            "--save-steps when omitted."
        ),
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help=(
            "Stop SFT early after this many consecutive checkpoint evaluations without "
            "meaningful improvement. When generated dev selection is active, both the "
            "generated dev metric and eval_loss must stop improving before training stops. "
            "Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--early-stopping-threshold",
        type=float,
        default=0.0,
        help="Minimum selection-metric improvement required to reset early-stopping patience.",
    )
    parser.add_argument(
        "--selection-metric",
        default="auto",
        choices=[
            "auto",
            "eval_loss",
            "generated_mean_f1",
            "generated_primary_score",
            "generated_macro_primary_score",
        ],
        help=(
            "Checkpoint-selection metric. 'auto' uses generated dev mean_f1 for "
            "list-only training and eval_loss otherwise."
        ),
    )
    parser.add_argument(
        "--generated-metric-subset-ids",
        default=None,
        help=(
            "Optional JSON file containing a list of question IDs (or a "
            "{question_ids: [...]} object). Generated-dev predictions are decoded once "
            "on the full eval set, then this subset receives an additional local MRR report."
        ),
    )
    parser.add_argument(
        "--selection-max-new-tokens",
        type=int,
        default=512,
        help="Max new tokens when generating dev predictions for checkpoint selection.",
    )
    parser.add_argument(
        "--selection-max-seq-length",
        type=int,
        default=None,
        help=(
            "Optional prompt-token cap for generated dev checkpoint selection. "
            "Defaults to the main --max-seq-length when omitted."
        ),
    )
    parser.add_argument(
        "--selection-num-generations",
        type=int,
        default=1,
        help="How many generations to sample per dev question during checkpoint selection.",
    )
    parser.add_argument(
        "--selection-aggregation-strategy",
        default="union",
        choices=["union", "frequency"],
        help="How to aggregate multiple dev generations during checkpoint selection.",
    )
    parser.add_argument(
        "--selection-aggregation-min-frequency",
        type=int,
        default=2,
        help="Minimum item frequency when using frequency aggregation for dev selection.",
    )
    parser.add_argument(
        "--selection-do-sample",
        action="store_true",
        help="Use sampling instead of greedy decoding for generated dev checkpoint selection.",
    )
    parser.add_argument(
        "--selection-temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for generated dev checkpoint selection.",
    )
    parser.add_argument(
        "--selection-top-p",
        type=float,
        default=0.9,
        help="Top-p sampling for generated dev checkpoint selection.",
    )
    parser.add_argument(
        "--summary-reference-mode",
        choices=["first", "all-mean", "all-max"],
        default="first",
        help="Legacy generation setting; official Phase-B checkpoint scoring excludes summary questions.",
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
            "Challenge format version passed to EvaluatorTask1b -e when official "
            "BioASQ scoring is used for generated dev selection."
        ),
    )
    parser.add_argument(
        "--bioasq-java-heap",
        default="4G",
        help="Java heap size passed to the official evaluator, for example 4G or 8G.",
    )
    parser.add_argument(
        "--dataset-num-proc",
        type=int,
        default=2,
        help="Number of processes for dataset mapping.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional trainer output directory. Defaults to a managed run folder.",
    )
    parser.add_argument(
        "--save-model-dir",
        default=None,
        help="Optional directory where the final adapter/tokenizer will be saved.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help=(
            "Optional checkpoint directory to resume from. Pass an explicit "
            "checkpoint path, 'latest' to require the newest checkpoint under "
            "--output-dir, or 'auto' to resume from the newest checkpoint when "
            "one exists and otherwise start fresh."
        ),
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional human-friendly label for the managed run folder.",
    )
    parser.add_argument(
        "--artifacts-root",
        default="Artifacts/models",
        help="Root directory for managed manifests and run folders.",
    )
    parser.add_argument(
        "--registry-path",
        default="models/registry.json",
        help="Repo-relative JSON registry used to track fine-tuning runs.",
    )
    parser.add_argument(
        "--register-alias",
        default=None,
        help="Optional alias to promote to this run after successful training.",
    )
    parser.set_defaults(**training_preset_defaults(preset_args.preset))
    return parser.parse_args()
