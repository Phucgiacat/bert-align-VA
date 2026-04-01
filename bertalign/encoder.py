import numpy as np
from sentence_transformers import SentenceTransformer
from bertalign.utils import yield_overlaps


class Encoder:
    """
    Sentence encoder wrapper supporting multiple multilingual models.
    
    For Vietnamese-English alignment, recommended models:
      - "intfloat/multilingual-e5-large"  (best quality)
      - "sentence-transformers/LaBSE"     (good, 109 langs)
      - "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
      - "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" (fastest)
    """

    def __init__(self, model_name):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        # multilingual-e5 models require a task prefix for best performance
        self._use_e5_prefix = "multilingual-e5" in model_name.lower()

    def _add_prefix(self, sentences, task="query"):
        """Add task prefix for E5 models (improves retrieval accuracy)."""
        prefix = f"{task}: "
        return [prefix + s for s in sentences]

    def transform(self, sents, num_overlaps):
        overlaps = list(yield_overlaps(sents, num_overlaps))

        # E5 models benefit from "passage:" prefix for encoding documents
        encode_input = self._add_prefix(overlaps, task="passage") if self._use_e5_prefix else overlaps

        sent_vecs = self.model.encode(
            encode_input,
            normalize_embeddings=True,   # cosine similarity via dot product
            show_progress_bar=False,
            batch_size=32,
        )

        embedding_dim = sent_vecs.size // (len(sents) * num_overlaps)
        sent_vecs = sent_vecs.copy()
        sent_vecs.resize(num_overlaps, len(sents), embedding_dim)

        len_vecs = [len(line.encode("utf-8")) for line in overlaps]
        len_vecs = np.array(len_vecs, dtype=np.float32)
        len_vecs.resize(num_overlaps, len(sents))

        return sent_vecs, len_vecs
