def target_cluster_count(llm, prompt) -> int:
    PROMPT: str = (
        f"According to the given prompt, please tell me how many cluster should I generate: {prompt}.\n\nJust give me the **pure number** without any explanation."
    )
    response, _, _ = llm.generate(PROMPT)
    try:
        return int(response.strip())
    except ValueError:
        print(f"Warning: Unable to parse cluster count from LLM response: {response}")
        return 0


def write_method(
    output_dir: str,
    method: str,
    evaluated: bool,
    model: str,
    dataset: str,
    notes: str = "",
) -> None:
    """Write experiment metadata to ``method.json`` in the output directory.

    Args:
        output_dir: Directory where ``method.json`` will be written.
        method: Clustering method used; one of ``'baseline'``, ``'bertopic'``,
            or ``'iac'``.
        evaluated: Whether the clustering has been evaluated against ground
            truth.
        model: LLM model name when *method* is ``'iac'``; ``'n/a'`` otherwise.
        dataset: Path to the dataset used for clustering.
        notes: Freeform notes field preserved across runs; defaults to ``''``.
    """
    import json
    import os

    data = {
        "method": method,
        "evaluated": evaluated,
        "model": model,
        "dataset": dataset,
        "notes": notes,
    }
    with open(os.path.join(output_dir, "method.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def mark_method_evaluated(output_dir: str) -> bool:
    """Mark an existing ``method.json`` as evaluated.

    Reads ``method.json`` from *output_dir*, sets ``evaluated`` to ``True``,
    and writes it back in place. No-ops silently if the file does not exist.

    Args:
        output_dir: Directory that may contain a ``method.json`` produced by a
            prior clustering run.

    Returns:
        ``True`` if ``method.json`` was found and updated, ``False`` if the
        file was not present.
    """
    import json
    import os

    path = os.path.join(output_dir, "method.json")
    if not os.path.exists(path):
        return False

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["evaluated"] = True

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return True


def get_output_files(output_dir: str) -> tuple[str, str, str, str]:
    import os

    log_file = os.path.join(output_dir, "log.txt")
    out_file = os.path.join(output_dir, "out.csv")
    summary_file = os.path.join(output_dir, "summary.json")
    sankey_file = os.path.join(output_dir, "sankey.json")

    return log_file, out_file, summary_file, sankey_file
