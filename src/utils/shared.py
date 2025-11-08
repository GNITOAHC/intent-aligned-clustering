def target_cluster_count(llm, prompt) -> int:
    PROMPT: str = f"According to the given prompt, please tell me how many cluster should I generate: {prompt}.\n\nJust give me the **pure number** without any explanation."
    response, _, _ = llm.generate(PROMPT)
    try:
        return int(response.strip())
    except ValueError:
        print(f"Warning: Unable to parse cluster count from LLM response: {response}")
        return 0


def get_output_files(output_dir: str) -> tuple[str, str, str]:
    import os

    log_file = os.path.join(output_dir, "log.txt")
    out_file = os.path.join(output_dir, "out.csv")
    summary_file = os.path.join(output_dir, "summary.json")

    return log_file, out_file, summary_file
