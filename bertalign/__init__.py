"""
Bertalign initialization - Vietnamese-English extended version
"""

__author__ = "Jason (bfsujason@163.com) | Extended for Vietnamese by Phucgiacat"
__version__ = "1.2.0"

from bertalign.encoder import Encoder

# Best multilingual models for Vietnamese-English alignment:
# 1. "BAAI/bge-m3"                            - SOTA! Top 1 currently for multilingual bitext mining & retrieval. (2.2GB)
# 2. "sentence-transformers/LaBSE"             - Trained explicitly for Bitext Mining by Google. Highly recommended for this task. (1.8GB)  
# 3. "intfloat/multilingual-e5-large"          - Good overall, but sometimes falls behind on native bitext matching without prefixes. (1.1GB)
# 4. "Alibaba-NLP/gte-multilingual-base"       - Faster but very strong.

DEFAULT_MODEL = "BAAI/bge-m3"

model_name = DEFAULT_MODEL
_model_instance = None

def get_model():
    global _model_instance
    if _model_instance is None:
        from bertalign.encoder import Encoder
        _model_instance = Encoder(DEFAULT_MODEL)
    return _model_instance

from bertalign.aligner import Bertalign
