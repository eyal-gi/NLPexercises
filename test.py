import re
import sys
import random
import math
import collections
import ex1


text = 'A cat sat on the mat. A fat cat sat on the mat. A rat sat on the mat. The rat sat on the cat. A bat spat on the rat that sat on the cat on the mat.'
nt = ex1.normalize_text(text)
print(nt)
lm = ex1.Ngram_Language_Model(n=3, chars=False)
lm.build_model(nt)  #*
# print(lm.get_model_dictionary())

# nt = re.sub('(?<! )(?=[.,!?()])|(?<=[.,!?()>])(?!abc)', r' ', "abc,")
lm.generate("on the")