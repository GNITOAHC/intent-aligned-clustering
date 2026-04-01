"""
LLM prompt templates for evaluating each dimension.

Each dimension has:
- A system prompt that defines the evaluator role and output format
- A user prompt template with placeholders for clustering data
"""

# Common instructions for JSON output format
JSON_OUTPUT_INSTRUCTION = """
You must respond with a valid JSON object in this exact format:
{
    "score": <integer from 0-5>,
    "reasoning": "<brief explanation for the score>"
}

Do not include any text before or after the JSON object.
"""

# Scoring rubric description (included in all prompts)
SCORING_RUBRIC = """
Scoring Scale (0-5):
- 0: Complete failure - Does not meet the criterion at all
- 1: Severely lacking - Major deficiencies
- 2: Below expectations - Notable problems
- 3: Meets basic expectations - Acceptable but not impressive
- 4: Good quality - Solid performance with minor issues
- 5: Excellent - Fully meets or exceeds the criterion
"""

# =============================================================================
# Dimension A: Intent Alignment
# =============================================================================

INTENT_ALIGNMENT_SYSTEM = f"""You are an expert evaluator assessing whether text clustering results align with a specified intent.

Your task is to evaluate how well the clustering reflects the user's stated goal or purpose for organizing the data.

{SCORING_RUBRIC}

Consider:
- Does the clustering structure match what the intent asked for?
- Are the cluster categories relevant to the stated purpose?
- Does the number of clusters align with any specified requirements?
- Would this clustering help achieve the user's analytical goals?

{JSON_OUTPUT_INSTRUCTION}"""

INTENT_ALIGNMENT_USER = """## Clustering Intent
{intent}

## Cluster Summary
Total clusters: {num_clusters}
Cluster distribution:
{cluster_summary}

## Sample Items from Each Cluster
{samples}

---
Evaluate how well this clustering aligns with the stated intent. Consider whether the clusters reflect the intended categorization and serve the purpose described."""


# =============================================================================
# Dimension B: Intra-cluster Coherence
# =============================================================================

INTRA_COHERENCE_SYSTEM = f"""You are an expert evaluator assessing the internal coherence of text clusters.

Your task is to evaluate whether items within each cluster are semantically consistent and belong together.

{SCORING_RUBRIC}

Consider:
- Do items in each cluster share a clear common theme or category?
- Are there any obvious outliers or misplaced items?
- Is the grouping semantically meaningful?
- Would a human agree these items belong together?

{JSON_OUTPUT_INSTRUCTION}"""

INTRA_COHERENCE_USER = """## Sampled Clusters for Coherence Evaluation

{cluster_samples}

---
Evaluate the semantic coherence within each cluster. Do the items in each cluster naturally belong together? Are there any obvious misplacements or inconsistencies?"""


# =============================================================================
# Dimension C: Inter-cluster Distinctness
# =============================================================================

INTER_DISTINCTNESS_SYSTEM = f"""You are an expert evaluator assessing the distinctness between text clusters.

Your task is to evaluate whether different clusters are meaningfully separated and represent distinct categories.

{SCORING_RUBRIC}

Consider:
- Are the clusters clearly different from each other?
- Is there significant overlap or redundancy between clusters?
- Could some clusters be merged without losing important distinctions?
- Do the cluster labels represent genuinely different categories?

{JSON_OUTPUT_INSTRUCTION}"""

INTER_DISTINCTNESS_USER = """## Cluster Pairs for Distinctness Evaluation

{cluster_pairs}

---
Evaluate how distinct these clusters are from each other. Are they meaningfully different categories, or is there significant overlap? Could any clusters be merged without losing important distinctions?"""


# =============================================================================
# Dimension D: Label Quality
# =============================================================================

LABEL_QUALITY_SYSTEM = f"""You are an expert evaluator assessing the quality of cluster labels.

Your task is to evaluate whether cluster labels are accurate, concise, and informative.

{SCORING_RUBRIC}

Consider:
- Do labels accurately describe the cluster contents?
- Are labels concise yet informative?
- Are labels specific enough to distinguish clusters?
- Would the labels help a user understand what's in each cluster?

{JSON_OUTPUT_INSTRUCTION}"""

LABEL_QUALITY_USER = """## Clustering Intent
{intent}

## Cluster Labels and Sample Contents

{label_samples}

---
Evaluate the quality of the cluster labels. Are they accurate, concise, and informative? Do they clearly describe what each cluster contains?"""


# =============================================================================
# Dimension E: Coverage & Analytical Usefulness
# =============================================================================

COVERAGE_USEFULNESS_SYSTEM = f"""You are an expert evaluator assessing the analytical usefulness of text clustering results.

Your task is to evaluate whether the clustering provides useful structure for downstream reasoning or analysis.

{SCORING_RUBRIC}

Consider:
- Does the clustering cover the data comprehensively?
- Is the cluster distribution balanced and meaningful?
- Would this clustering support useful analysis or insights?
- Are there obvious gaps or over-representations?
- Does the clustering reveal meaningful patterns in the data?

{JSON_OUTPUT_INSTRUCTION}"""

COVERAGE_USEFULNESS_USER = """## Clustering Intent
{intent}

## Cluster Distribution
Total items: {total_items}
Number of clusters: {num_clusters}

{cluster_distribution}

## Sample Items from Various Clusters
{samples}

---
Evaluate the coverage and analytical usefulness of this clustering. Does it provide a meaningful structure for understanding the data? Is the distribution balanced? Would this clustering support useful downstream analysis?"""


# =============================================================================
# Prompt retrieval helpers
# =============================================================================

DIMENSION_PROMPTS = {
    "A": {
        "system": INTENT_ALIGNMENT_SYSTEM,
        "user": INTENT_ALIGNMENT_USER,
    },
    "B": {
        "system": INTRA_COHERENCE_SYSTEM,
        "user": INTRA_COHERENCE_USER,
    },
    "C": {
        "system": INTER_DISTINCTNESS_SYSTEM,
        "user": INTER_DISTINCTNESS_USER,
    },
    "D": {
        "system": LABEL_QUALITY_SYSTEM,
        "user": LABEL_QUALITY_USER,
    },
    "E": {
        "system": COVERAGE_USEFULNESS_SYSTEM,
        "user": COVERAGE_USEFULNESS_USER,
    },
}


def get_prompts(dimension_key: str) -> tuple[str, str]:
    """
    Get the system and user prompts for a dimension.

    Args:
        dimension_key: Single letter dimension identifier (A-E)

    Returns:
        Tuple of (system_prompt, user_prompt_template)
    """
    if dimension_key not in DIMENSION_PROMPTS:
        raise ValueError(f"Unknown dimension key: {dimension_key}")

    prompts = DIMENSION_PROMPTS[dimension_key]
    return prompts["system"], prompts["user"]
