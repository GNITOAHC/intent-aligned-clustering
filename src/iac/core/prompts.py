def create_init_cluster_with_sample_prompt(instruction, sample_texts) -> str:
    init_prompt = f"""CLUSTER INITIALIZATION TASK:

Objective: {instruction}

Sample documents:
{chr(10).join(sample_texts)}

Based on these sample documents and the clustering objective, suggest 3-5 initial cluster labels that would be useful for categorizing similar documents.

Rules:
1. Labels should be 1-3 words each
2. Labels should directly relate to the objective: {instruction}
3. Labels should capture distinct categories visible in the samples
4. Avoid overly specific or overly broad categories

Return only the cluster labels, one per line, no explanations:"""
    return init_prompt


def create_cluster_prompt_sys() -> str:
    return (
        "You are an expert document classifier specializing in semantic clustering. "
        "Your task is to assign the most appropriate cluster label based on the document's core intent and meaning. "
        "CRITICAL RULES:\n"
        "1. STRONGLY prefer existing labels when they semantically match (even if not perfect)\n"
        "2. Only create NEW labels if NO existing label captures the document's essence\n"
        "3. New labels must be concise (1-3 words), descriptive, and directly aligned with the clustering instruction\n"
        "4. Focus on the document's PRIMARY intent, not minor details or tangential topics\n"
        "5. Return ONLY the label name - no explanations, quotes, or extra text"
    )


def create_cluster_prompt(cluster_types: list[str], instruction: str, text: str) -> str:
    cluster_list = (
        ", ".join(f'"{label}"' for label in cluster_types)
        if cluster_types
        else "None yet defined"
    )

    return f"""CLUSTERING TASK:
Objective: {instruction}

EXISTING CLUSTERS: {cluster_list}

DOCUMENT TO CLASSIFY:
---
{text}
---

CLASSIFICATION INSTRUCTIONS:
1. Analyze the document's PRIMARY purpose and meaning in relation to: {instruction}
2. If ANY existing cluster reasonably captures this document's essence, USE IT (exact match required)
3. Only if NO existing cluster fits, create a NEW descriptive label that:
   - Is 1-3 words maximum
   - Directly relates to the clustering objective
   - Captures the document's main intent/category
4. Consider semantic similarity, not just keyword matching

RESPONSE FORMAT: Return only the cluster label (no quotes, explanations, or additional text)

CLUSTER LABEL:"""


def create_cluster_refinement_prompt(
    clusters: dict[str, list], instruction: str
) -> str:
    """Create a prompt for refining cluster labels to be more coherent"""
    cluster_info = []
    for label, doc_count in clusters.items():
        cluster_info.append(f'"{label}" ({len(doc_count)} documents)')

    return f"""CLUSTER REFINEMENT TASK:

Original clustering objective: {instruction}

Current clusters: {", ".join(cluster_info)}

TASK: Review these cluster labels for consistency and clarity. Suggest better names if needed.
Rules:
1. Labels should be concise (1-3 words)
2. Labels should clearly reflect the clustering objective
3. Avoid redundant or overly similar labels
4. Use clear, descriptive terms

Respond with: KEEP (if good) or RENAME_TO: new_name for each cluster, one per line.
Format: "old_label" -> KEEP or "old_label" -> RENAME_TO: new_label

Response:"""
