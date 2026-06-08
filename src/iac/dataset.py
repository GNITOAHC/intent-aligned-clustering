import os


class IACDataset:
    """
    Intent Aligned Clustering Dataset
    This class stores the actual text and it's metadata, including it's embedding, tags and other information.
    """

    metadata: list[dict]
    texts: list[str]
    rewrites: list[str]  # placeholder

    def __init__(self, metadata: list[dict], text: list[str]):
        self.metadata = metadata
        self.texts = text

    def __getitem__(self, index) -> tuple[dict, str]:
        return self.metadata[index], self.texts[index]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.texts)

    @classmethod
    def from_list(cls, metadata: list[dict], text: list[str]):
        return cls(metadata=metadata, text=text)

    @classmethod
    def from_dir(cls, path: str):
        """
        Load all text files from a directory.
        metadata = {"filename": <name>, "path": <fullpath>}
        texts = file content
        """
        metadata, texts = [], []
        for fname in sorted(os.listdir(path)):
            fpath = os.path.join(path, fname)
            if os.path.isfile(fpath) and fname.lower().endswith(".txt"):
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                metadata.append({"filename": fname, "path": fpath})
                texts.append(content)
        return cls(metadata=metadata, text=texts)

    @classmethod
    def from_csv(cls, path: str):
        """
        Load dataset from CSV.
        By default assumes a column named "text".
        All other columns are stored as metadata.
        """
        import csv

        metadata, texts = [], []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV is empty")
            if "text" not in reader.fieldnames:
                raise ValueError("CSV must contain a 'text' column")

            for row in reader:
                text = row.pop("text")
                metadata.append(row)
                texts.append(text)
        return cls(metadata=metadata, text=texts)
