import sys
sys.path.append("..")
import numpy as np

from my_nlp.common.layers import MatMul
from my_nlp.common.util import preprocess, create_contexts_target, convert_one_hot

text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word= preprocess(text)
print(corpus)

print(id_to_word)

contexts, target = create_contexts_target(corpus)
print(contexts)


vocab_size = len(word_to_id)
target = convert_one_hot(target, vocab_size)
print(target)
