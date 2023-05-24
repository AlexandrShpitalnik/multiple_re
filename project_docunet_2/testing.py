from src.transformers import (
    XLMRobertaConfig,
    XLMRobertaTokenizer,
    BertModel
)

from tokenizers.implementations import BertWordPieceTokenizer

if __name__ == '__main__':
    cls = BertModel(None)