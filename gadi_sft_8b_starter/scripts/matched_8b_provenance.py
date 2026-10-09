"""Compare matched 8B provenance across the legacy/default-mode schema change."""


def normalise_provenance(provenance):
    # Before execution-mode controls existed, all matched runs used default.
    # Default only this known legacy omission; retain every other field/hash.
    return {name: {**source, "execution_mode": source.get("execution_mode", "default")}
            for name, source in provenance.items()}


def differing_fields(first, second, prefix=""):
    if isinstance(first, dict) and isinstance(second, dict):
        result = []
        for key in sorted(first.keys() | second.keys()):
            path = f"{prefix}.{key}" if prefix else key
            if key not in first or key not in second:
                result.append(path)
            else:
                result.extend(differing_fields(first[key], second[key], path))
        return result
    return [] if first == second else [prefix]


def validate_provenance(previous, current):
    differences = differing_fields(normalise_provenance(previous), normalise_provenance(current))
    if differences:
        raise ValueError("Baseline provenance differs in: " + ", ".join(differences))
