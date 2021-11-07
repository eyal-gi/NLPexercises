import re
import sys
import random
import math
import collections
# random.seed(1489)
import ex1


text = 'A cat sat on the mat. A fat cat sat on the mat. A rat sat on the mat. The rat sat on the cat. A bat spat on the rat that sat on the cat on the mat.'
# text = open('big.txt').read()
# text = 'a cat sat a rat sat a bat sat'
nt = ex1.normalize_text(text)
print(nt)
lm = ex1.Ngram_Language_Model(n=5, chars=True)
lm.build_model(nt)  #*
print(lm.get_model_dictionary())

# nt = re.sub('(?<! )(?=[.,!?()])|(?<=[.,!?()>])(?!abc)', r' ', "abc,")
t = lm.generate('a ca', n=30)
print(t)
# print(lm.evaluate(t))
# print(lm.evaluate("a cat sat on the mat . a fat cat sat on the mat . a bat spat on the mat . a rat sat on the mat ."))
# print(lm.evaluate("p a cat eyal ginosar gever on the mat"))
# print(lm.evaluate(""))
# print(lm.evaluate('the rat sat on the cat'))
# # print(lm.get_model_dictionary())
# n=3
# str = 'a cat'
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
