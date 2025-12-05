def create_labeling_prompt(instruction: str, text: str) -> str:
    return f"""You are an expert document classifier.
Your task is to assign a single, concise label to the following document based on the given instruction.

INSTRUCTION: {instruction}

DOCUMENT:
---
{text}
---

REQUIREMENTS:
1. Return ONLY the label.
2. The label should be concise (1-4 words).
3. Do not include any explanation or punctuation.
4. If the document is irrelevant to the instruction, return "Other".

LABEL:"""
