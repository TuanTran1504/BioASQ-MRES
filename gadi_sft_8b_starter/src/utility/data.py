from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.model_registry import get_project_root, resolve_repo_path
from src.prompt_registry import resolve_prompt_bundle

from .config import QUESTION_INSTRUCTIONS


Record = Dict[str, Any]
INPUT_FIELD_PATTERN = re.compile(r"input_(\d+)$")


def clean_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def clean_multiline_text(text: Any) -> str:
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"\s+", " ", line).strip() for line in normalized.split("\n")]
    return "\n".join(line for line in lines if line)


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 18].rstrip() + " ... [TRUNCATED]"


def list_record_resources(record: Record) -> List[str]:
    resources: List[Tuple[int, str]] = []
    for key, value in record.items():
        match = INPUT_FIELD_PATTERN.fullmatch(str(key))
        if match is None:
            continue
        field_index = int(match.group(1))
        if field_index < 2:
            continue
        cleaned_value = clean_multiline_text(value)
        if cleaned_value:
            resources.append((field_index, cleaned_value))
    resources.sort(key=lambda item: item[0])
    return [value for _, value in resources]


def build_unitor_prompt_text(
    instruction: str,
    question: str,
    resources: Sequence[str],
    answer: Optional[str] = None,
    eos_token: str = "",
) -> str:
    prompt = f"{clean_multiline_text(instruction)}\n\n# Question: {clean_text(question)}\n# PubMed resources:"
    resource_lines = [clean_multiline_text(resource) for resource in resources if clean_multiline_text(resource)]
    if resource_lines:
        prompt = f"{prompt} {resource_lines[0]}"
        if len(resource_lines) > 1:
            prompt = f"{prompt}\n" + "\n".join(resource_lines[1:])

    if answer is None:
        return f"{prompt}\n# Answer:"

    answer_text = clean_text(answer)
    eos_suffix = f" {eos_token}" if eos_token else ""
    return f"{prompt}\n# Answer: {answer_text}{eos_suffix}"


def extract_pmid(document_url: str) -> str:
    match = re.search(r"/pubmed/([^/?#]+)", document_url or "")
    return match.group(1) if match else ""


def serialize_resource(document_url: str, snippets: Sequence[Record], max_chars: int) -> str:
    pmid = extract_pmid(document_url)
    header = f"PubMed ID: {pmid}" if pmid else f"Document: {document_url}"

    parts: List[str] = [header]
    for snippet in snippets:
        text = clean_text(snippet.get("text", ""))
        if not text:
            continue

        begin_section = clean_text(snippet.get("beginSection", ""))
        end_section = clean_text(snippet.get("endSection", ""))
        section = ""
        if begin_section and end_section and begin_section != end_section:
            section = f" ({begin_section} -> {end_section})"
        elif begin_section:
            section = f" ({begin_section})"

        parts.append(f"[BS]{text}[ES]{section}")

    return truncate_text("\n".join(parts), max_chars=max_chars)


class EmbeddingResourceSelector:
    def __init__(
        self,
        model_name: str,
        *,
        article_model_name: str | None = None,
        device: str = "auto",
        batch_size: int = 32,
        local_files_only: bool = False,
    ) -> None:
        self.model_name = clean_text(model_name)
        self.article_model_name = clean_text(article_model_name)
        self.device = clean_text(device) or "auto"
        self.batch_size = max(1, int(batch_size))
        self.local_files_only = bool(local_files_only)
        self._model: Any = None
        self._query_tokenizer: Any = None
        self._query_model: Any = None
        self._article_tokenizer: Any = None
        self._article_model: Any = None

    def _resolved_article_model_name(self) -> str:
        if self.article_model_name:
            return self.article_model_name
        if self.model_name == "ncbi/MedCPT-Query-Encoder":
            return "ncbi/MedCPT-Article-Encoder"
        return self.model_name

    def _uses_dual_transformer_encoders(self) -> bool:
        article_model_name = self._resolved_article_model_name()
        return bool(article_model_name and article_model_name != self.model_name)

    def _query_max_length(self) -> int:
        return 64 if self.model_name == "ncbi/MedCPT-Query-Encoder" else 512

    def _article_max_length(self) -> int:
        return 512

    def _resolve_device(self) -> str:
        normalized_device = clean_text(self.device).lower() or "auto"
        if normalized_device == "conda":
            normalized_device = "cuda"
        if normalized_device == "gpu":
            normalized_device = "cuda"
        if normalized_device == "auto":
            try:
                import torch
            except ImportError:
                return "cpu"
            return "cuda" if torch.cuda.is_available() else "cpu"

        try:
            import torch

            return str(torch.device(normalized_device))
        except Exception as exc:
            raise ValueError(
                f"Invalid embedding reranker device '{self.device}'. "
                "Use 'cpu', 'cuda', or 'auto'."
            ) from exc

    def _load_model(self) -> Any:
        if self._uses_dual_transformer_encoders():
            raise RuntimeError(
                "Dual-encoder resource selection should use _load_dual_transformer_models()."
            )
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise ImportError(
                "Embedding-based resource selection requires sentence-transformers. "
                "Install it in the evaluation environment first."
            ) from exc

        init_kwargs = {"device": self._resolve_device()}
        try:
            self._model = SentenceTransformer(
                self.model_name,
                local_files_only=self.local_files_only,
                **init_kwargs,
            )
        except TypeError:
            self._model = SentenceTransformer(self.model_name, **init_kwargs)
        except OSError as exc:
            locality_hint = (
                "The model is not available in the local Hugging Face cache. "
                "Run once with network access, or pre-download it before using "
                "--local-files-only."
                if self.local_files_only
                else "The model could not be downloaded or loaded from cache."
            )
            raise OSError(
                f"Could not load embedding reranker model '{self.model_name}'. {locality_hint}"
            ) from exc
        return self._model

    def _load_dual_transformer_models(self) -> tuple[Any, Any, Any, Any]:
        if (
            self._query_tokenizer is not None
            and self._query_model is not None
            and self._article_tokenizer is not None
            and self._article_model is not None
        ):
            return (
                self._query_tokenizer,
                self._query_model,
                self._article_tokenizer,
                self._article_model,
            )

        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise ImportError(
                "Dual-encoder resource selection requires transformers. "
                "Install it in the evaluation environment first."
            ) from exc

        import torch

        device_name = self._resolve_device()
        torch_device = torch.device(device_name)
        query_model_name = self.model_name
        article_model_name = self._resolved_article_model_name()
        try:
            self._query_tokenizer = AutoTokenizer.from_pretrained(
                query_model_name,
                local_files_only=self.local_files_only,
                trust_remote_code=True,
            )
            self._query_model = AutoModel.from_pretrained(
                query_model_name,
                local_files_only=self.local_files_only,
                trust_remote_code=True,
            ).to(torch_device).eval()
            self._article_tokenizer = AutoTokenizer.from_pretrained(
                article_model_name,
                local_files_only=self.local_files_only,
                trust_remote_code=True,
            )
            self._article_model = AutoModel.from_pretrained(
                article_model_name,
                local_files_only=self.local_files_only,
                trust_remote_code=True,
            ).to(torch_device).eval()
        except OSError as exc:
            locality_hint = (
                "The model is not available in the local Hugging Face cache. "
                "Run once with network access, or pre-download it before using "
                "--local-files-only."
                if self.local_files_only
                else "The model could not be downloaded or loaded from cache."
            )
            raise OSError(
                "Could not load dual-encoder resource reranker models "
                f"'{query_model_name}' and '{article_model_name}'. {locality_hint}"
            ) from exc
        return (
            self._query_tokenizer,
            self._query_model,
            self._article_tokenizer,
            self._article_model,
        )

    def _prepare_candidate_text(self, text: str) -> str:
        cleaned = clean_multiline_text(text)
        if not cleaned:
            return ""

        lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
        if lines and (lines[0].startswith("PubMed ID:") or lines[0].startswith("Document:")):
            lines = lines[1:]

        if not lines:
            return cleaned

        normalized = "\n".join(lines)
        normalized = normalized.replace("[BS]", "").replace("[ES]", "")
        normalized = re.sub(r"\s+\((?:abstract|title|methods|results|conclusion|conclusions|background|objective|objectives|discussion|introduction|case report|review|summary)(?:\s*->\s*[^)]*)?\)", "", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized or cleaned

    def _encode_dual_transformer_texts(
        self,
        *,
        texts: Sequence[str],
        tokenizer: Any,
        model: Any,
        max_length: int,
    ) -> Any:
        import torch
        import torch.nn.functional as F

        device = next(model.parameters()).device
        encoded = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=max_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            last_hidden_state = model(**encoded).last_hidden_state
            embeddings = last_hidden_state[:, 0, :]
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def _rank_with_dual_transformers(self, query: str, candidates: Sequence[str]) -> List[int]:
        import torch

        indexed_candidates = [
            (index, self._prepare_candidate_text(candidate))
            for index, candidate in enumerate(candidates)
            if self._prepare_candidate_text(candidate)
        ]
        if not indexed_candidates:
            return []

        (
            query_tokenizer,
            query_model,
            article_tokenizer,
            article_model,
        ) = self._load_dual_transformer_models()
        query_embedding = self._encode_dual_transformer_texts(
            texts=[query],
            tokenizer=query_tokenizer,
            model=query_model,
            max_length=self._query_max_length(),
        )[0]
        candidate_embeddings = self._encode_dual_transformer_texts(
            texts=[candidate_text for _, candidate_text in indexed_candidates],
            tokenizer=article_tokenizer,
            model=article_model,
            max_length=self._article_max_length(),
        )
        scores = torch.matmul(candidate_embeddings, query_embedding).detach().cpu().tolist()
        ranked_pairs = sorted(
            zip(indexed_candidates, scores),
            key=lambda item: (-float(item[1]), item[0][0]),
        )
        return [index for (index, _), _score in ranked_pairs]

    def rank(self, query: str, candidates: Sequence[str]) -> List[int]:
        normalized_query = clean_text(query)
        if not normalized_query:
            return list(range(len(candidates)))

        if self._uses_dual_transformer_encoders():
            return self._rank_with_dual_transformers(normalized_query, candidates)

        indexed_candidates = [
            (index, self._prepare_candidate_text(candidate))
            for index, candidate in enumerate(candidates)
            if self._prepare_candidate_text(candidate)
        ]
        if not indexed_candidates:
            return []

        model = self._load_model()
        candidate_texts = [candidate_text for _, candidate_text in indexed_candidates]
        query_embedding = model.encode(
            [normalized_query],
            batch_size=1,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        candidate_embeddings = model.encode(
            candidate_texts,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        scores = (candidate_embeddings @ query_embedding).detach().cpu().tolist()
        ranked_pairs = sorted(
            zip(indexed_candidates, scores),
            key=lambda item: (-float(item[1]), item[0][0]),
        )
        return [index for (index, _), _score in ranked_pairs]


def build_document_resource_candidates(question: Record, max_resource_chars: int) -> List[str]:
    grouped: Dict[str, List[Record]] = {}
    ordered_docs: List[str] = []

    for snippet in question.get("snippets", []):
        if not isinstance(snippet, dict):
            continue

        document_url = clean_text(snippet.get("document", ""))
        if not document_url:
            document_url = f"document_{len(ordered_docs) + 1}"

        if document_url not in grouped:
            grouped[document_url] = []
            ordered_docs.append(document_url)

        grouped[document_url].append(snippet)

    return [
        serialize_resource(doc_url, grouped[doc_url], max_chars=max_resource_chars)
        for doc_url in ordered_docs
    ]


def build_snippet_resource_candidates(question: Record, max_resource_chars: int) -> List[str]:
    resources: List[str] = []
    for index, snippet in enumerate(question.get("snippets", []), start=1):
        if not isinstance(snippet, dict):
            continue
        document_url = clean_text(snippet.get("document", "")) or f"document_{index}"
        resources.append(serialize_resource(document_url, [snippet], max_chars=max_resource_chars))
    return resources


def select_resource_texts(
    resources: Sequence[str],
    *,
    question_text: str,
    max_resources: int,
    resource_selection: str = "first",
    resource_selector: Optional[EmbeddingResourceSelector] = None,
) -> List[str]:
    selected_resources = [clean_multiline_text(resource) for resource in resources if clean_multiline_text(resource)]
    normalized_selection = clean_text(resource_selection).lower() or "first"
    if normalized_selection not in {"first", "embedding"}:
        raise ValueError(f"Unsupported resource selection mode: {resource_selection}")

    if normalized_selection == "embedding" and selected_resources:
        if resource_selector is None:
            raise ValueError(
                "Embedding-based resource selection was requested, but no embedding selector was provided."
            )
        ranked_indices = resource_selector.rank(question_text, selected_resources)
        selected_resources = [selected_resources[index] for index in ranked_indices]

    if max_resources > 0:
        selected_resources = selected_resources[:max_resources]
        while len(selected_resources) < max_resources:
            selected_resources.append("")

    return selected_resources


def build_resources(
    question: Record,
    max_resources: int,
    max_resource_chars: int,
    *,
    question_text: str = "",
    resource_granularity: str = "document",
    resource_selection: str = "first",
    resource_selector: Optional[EmbeddingResourceSelector] = None,
) -> List[str]:
    normalized_granularity = clean_text(resource_granularity).lower() or "document"
    if normalized_granularity == "document":
        resource_candidates = build_document_resource_candidates(question, max_resource_chars=max_resource_chars)
    elif normalized_granularity == "snippet":
        resource_candidates = build_snippet_resource_candidates(question, max_resource_chars=max_resource_chars)
    else:
        raise ValueError(f"Unsupported resource granularity: {resource_granularity}")

    resources = select_resource_texts(
        resource_candidates,
        question_text=question_text or clean_text(question.get("body", "")),
        max_resources=max_resources,
        resource_selection=resource_selection,
        resource_selector=resource_selector,
    )

    while max_resources > 0 and len(resources) < max_resources:
        resources.append("")

    return resources


def normalize_summary_output(question: Record, max_summary_answers: int) -> str:
    ideal_answers = question.get("ideal_answer", [])
    if isinstance(ideal_answers, str):
        return clean_text(ideal_answers)
    if not isinstance(ideal_answers, list):
        return ""

    chosen = [clean_text(answer) for answer in ideal_answers if clean_text(answer)]
    return "\n".join(chosen[:max_summary_answers]).strip()


def normalize_factoid_output(question: Record, max_factoid_answers: int) -> str:
    exact_answer = question.get("exact_answer")
    expressions: List[str] = []

    if isinstance(exact_answer, str):
        expressions = [clean_text(exact_answer)]
    elif isinstance(exact_answer, list):
        for item in exact_answer:
            if isinstance(item, list):
                for alias in item:
                    alias = clean_text(alias)
                    if alias:
                        expressions.append(alias)
            else:
                item = clean_text(item)
                if item:
                    expressions.append(item)

    expressions = [value for value in expressions if value][:max_factoid_answers]
    return " ".join(f"[BE]{value}[EE]" for value in expressions)


def normalize_list_output(question: Record, max_list_items: int) -> str:
    exact_answer = question.get("exact_answer")
    items: List[str] = []

    if isinstance(exact_answer, list):
        for item in exact_answer:
            if isinstance(item, list):
                canonical = next((clean_text(alias) for alias in item if clean_text(alias)), "")
                if canonical:
                    items.append(canonical)
            else:
                item = clean_text(item)
                if item:
                    items.append(item)

    items = items[:max_list_items]
    return " ".join(f"[BI]{value}[EI]" for value in items)


def normalize_yesno_output(question: Record) -> str:
    exact_answer = clean_text(question.get("exact_answer", "")).lower()
    if exact_answer in {"yes", "no"}:
        return exact_answer

    ideal_answer = question.get("ideal_answer", "")
    if isinstance(ideal_answer, list):
        ideal_text = clean_text(next((item for item in ideal_answer if clean_text(item)), ""))
    else:
        ideal_text = clean_text(ideal_answer)

    if ideal_text.lower().startswith("yes"):
        return "yes"
    if ideal_text.lower().startswith("no"):
        return "no"
    return ""


def build_output(question: Record, args: argparse.Namespace) -> str:
    qtype = clean_text(question.get("type", "")).lower()
    if qtype == "summary":
        return normalize_summary_output(question, max_summary_answers=args.max_summary_answers)
    if qtype == "factoid":
        return normalize_factoid_output(question, max_factoid_answers=args.max_factoid_answers)
    if qtype == "list":
        return normalize_list_output(question, max_list_items=args.max_list_items)
    if qtype == "yesno":
        return normalize_yesno_output(question)
    return ""


def resolve_prompt_instructions(args: argparse.Namespace) -> Dict[str, str]:
    project_root = get_project_root()
    prompt_path_value = getattr(args, "prompt_file", None) or getattr(args, "prompt_registry_path", None)
    prompt_registry_path = resolve_repo_path(prompt_path_value, project_root=project_root)
    prompt_bundle = resolve_prompt_bundle(
        registry_path=prompt_registry_path,
        prompt_ref=getattr(args, "prompt", None),
        fallback_instructions=QUESTION_INSTRUCTIONS,
    )
    return {
        question_type: clean_multiline_text(instruction)
        for question_type, instruction in prompt_bundle.get("instructions", {}).items()
        if clean_multiline_text(instruction)
    }


def convert_bioasq_question(
    question: Record,
    args: argparse.Namespace,
    prompt_instructions: Optional[Dict[str, str]] = None,
) -> Optional[Record]:
    qtype = clean_text(question.get("type", "")).lower()
    if qtype not in set(args.question_types):
        return None

    prompt_instructions = prompt_instructions or QUESTION_INSTRUCTIONS
    instruction = prompt_instructions.get(qtype) or QUESTION_INSTRUCTIONS.get(qtype)
    question_text = clean_text(question.get("body", ""))
    if not instruction or not question_text:
        return None

    resources = build_resources(
        question,
        max_resources=args.max_resources,
        max_resource_chars=args.max_resource_chars,
        question_text=question_text,
        resource_granularity=getattr(args, "resource_granularity", "document"),
        resource_selection=getattr(args, "resource_selection", "first"),
    )
    output = build_output(question, args)
    if not output:
        return None

    record: Record = {
        "id": clean_text(question.get("id", "")),
        "type": qtype,
        "instruction": instruction,
        "input_1": question_text,
        "output": output,
    }
    for index, resource in enumerate(resources, start=2):
        record[f"input_{index}"] = resource
    return record


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_prepared_records(path: Path, prompt_instructions: Optional[Dict[str, str]] = None) -> List[Record]:
    data = read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Prepared dataset must be a JSON list: {path}")

    required = {"instruction", "input_1", "output", "type"}
    rows: List[Record] = []
    for row in data:
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError(f"Prepared dataset row is missing required fields in {path}")
        row_type = clean_text(row.get("type", "")).lower()
        instruction_override = (
            clean_multiline_text(prompt_instructions.get(row_type, ""))
            if prompt_instructions
            else ""
        )
        rows.append(
            {
                "id": clean_text(row.get("id", "")),
                "type": row_type,
                "instruction": instruction_override or clean_multiline_text(row.get("instruction", "")),
                "input_1": clean_text(row.get("input_1", "")),
                "input_2": clean_multiline_text(row.get("input_2", "")),
                "input_3": clean_multiline_text(row.get("input_3", "")),
                "input_4": clean_multiline_text(row.get("input_4", "")),
                "output": clean_multiline_text(row.get("output", "")),
            }
        )
        for key, value in row.items():
            match = INPUT_FIELD_PATTERN.fullmatch(str(key))
            if match is None:
                continue
            field_index = int(match.group(1))
            if field_index <= 4:
                continue
            rows[-1][f"input_{field_index}"] = clean_multiline_text(value)
    return rows


def load_bioasq_records(
    path: Path,
    args: argparse.Namespace,
    prompt_instructions: Optional[Dict[str, str]] = None,
) -> List[Record]:
    data = read_json(path)
    questions = data.get("questions")
    if not isinstance(questions, list):
        raise ValueError(f"Raw BioASQ file must contain a 'questions' list: {path}")

    rows: List[Record] = []
    for question in questions:
        if not isinstance(question, dict):
            continue
        converted = convert_bioasq_question(question, args, prompt_instructions=prompt_instructions)
        if converted is not None:
            rows.append(converted)
    return rows


def load_records(
    paths: Sequence[str],
    args: argparse.Namespace,
    prompt_instructions: Optional[Dict[str, str]] = None,
) -> List[Record]:
    all_rows: List[Record] = []
    seen_ids = set()
    allowed_types = {clean_text(qtype).lower() for qtype in args.question_types}

    for raw_path in paths:
        path = Path(raw_path)
        data = read_json(path)

        if isinstance(data, list):
            rows = load_prepared_records(path, prompt_instructions=prompt_instructions)
        elif isinstance(data, dict) and "questions" in data:
            rows = load_bioasq_records(path, args, prompt_instructions=prompt_instructions)
        else:
            raise ValueError(f"Unsupported input format for {path}")

        for row in rows:
            row_type = clean_text(row.get("type", "")).lower()
            if row_type not in allowed_types:
                continue
            row_id = row.get("id") or f"{path.name}:{len(all_rows)}"
            dedupe_key = (row_id, row_type, row.get("output", ""))
            if dedupe_key in seen_ids:
                continue
            seen_ids.add(dedupe_key)
            all_rows.append(row)

    return all_rows


def maybe_cap_rows(rows: List[Record], limit: Optional[int]) -> List[Record]:
    return rows if limit is None else rows[:limit]


def save_prepared_records(rows: Sequence[Record], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(list(rows), handle, ensure_ascii=False, indent=2)


def split_train_eval(
    rows: List[Record],
    validation_ratio: float,
    seed: int,
) -> Tuple[List[Record], List[Record]]:
    from sklearn.model_selection import train_test_split

    if validation_ratio <= 0:
        return rows, []

    if len(rows) < 2:
        raise ValueError("Need at least 2 records to create a train/eval split.")

    labels = [row["type"] for row in rows]
    unique_labels = set(labels)
    stratify = labels if len(unique_labels) > 1 and all(labels.count(label) > 1 for label in unique_labels) else None

    train_rows, eval_rows = train_test_split(
        rows,
        test_size=validation_ratio,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return train_rows, eval_rows


def prepare_train_eval_rows(args: argparse.Namespace) -> Tuple[List[Record], List[Record]]:
    prompt_instructions = resolve_prompt_instructions(args)
    train_rows = load_records(args.train_input, args, prompt_instructions=prompt_instructions)
    if not train_rows:
        raise ValueError("No training examples were prepared from train-input.")

    if args.eval_input:
        eval_rows = load_records(args.eval_input, args, prompt_instructions=prompt_instructions)
    else:
        train_rows, eval_rows = split_train_eval(
            train_rows,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
        )

    train_rows = maybe_cap_rows(train_rows, args.max_train_samples)
    eval_rows = maybe_cap_rows(eval_rows, args.max_eval_samples)
    return train_rows, eval_rows
