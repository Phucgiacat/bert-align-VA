from setuptools import setup, find_packages

setup(
    name='bertalign-vi',
    version='1.2.0',
    url='https://github.com/Phucgiacat/bert-align-VA.git',
    description='An automatic multilingual sentence aligner (Extended for Vietnamese-English).',
    packages=find_packages(),    
    install_requires=[
        'numba',
        'faiss-cpu',
        'langdetect',
        'sentence-splitter>=1.4',
        'sentence-transformers',
        'transformers',
        'sentencepiece',
        'protobuf',
        'pyvi',
        'underthesea',
    ],
)