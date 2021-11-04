import re
import sys
import random
import math
import collections
import ex1


text = 'A cat sat on the mat. A fat cat sat on the mat. A rat sat on the mat. The rat sat on the cat. A bat spat on the rat that sat on the cat on the mat.'
# text = open('big.txt').read()
nt = ex1.normalize_text(text)
# print(nt)
lm = ex1.Ngram_Language_Model(n=3, chars=False)
lm.build_model(nt)  #*
# print(lm.get_model_dictionary())

# nt = re.sub('(?<! )(?=[.,!?()])|(?<=[.,!?()>])(?!abc)', r' ', "abc,")
lm.generate('a cat')
# print(lm.evaluate(lm.generate('a cat', n=30)))
print(lm.evaluate("a cat sat on the mat"))
print(lm.get_model_dictionary())
# print(lm.evaluate('the rat sat on the cat .'))
n=3
str = 'a cat'
# new = str.split(' ')
#
# new = new[len(new)-(n-1):]
# new1 = ' '.join(new)
# print(new1)
# print(str.split(' '))
# print(str.split(' ', 1)[1])
# context = str
# ngram = context.split(' ')
# ngram = ' '.join(ngram[len(ngram) - (n - 1):])
# print(ngram)