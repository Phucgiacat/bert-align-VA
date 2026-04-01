"""
Bertalign initialization - Vietnamese-English extended version
"""

__author__ = "Jason (bfsujason@163.com) | Extended for Vietnamese by Phucgiacat"
__version__ = "1.2.0"

from bertalign.encoder import Encoder

# Best multilingual models for Vietnamese-English alignment:
# 1. "intfloat/multilingual-e5-large"          - State-of-the-art, best quality (1.1GB)
# 2. "sentence-transformers/LaBSE"             - Supports 109 langs incl. Vietnamese (1.8GB)  
# 3. "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" - Lighter option (1.1GB)
# 4. "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" - Fastest, lighter (470MB)
#
# For Vietnamese-English, "intfloat/multilingual-e5-large" gives the best performance.
# Change DEFAULT_MODEL to switch between models.

DEFAULT_MODEL = "intfloat/multilingual-e5-large"

model_name = DEFAULT_MODEL
model = Encoder(model_name)

from bertalign.aligner import Bertalign
