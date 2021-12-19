import re
import sys
import random
import math
import collections
import ex1
import ex2
import spelling_confusion_matrices

text =open('big.txt').read()
nt = ex2.normalize_text(text)

lm = ex1.Ngram_Language_Model(3)
lm.build_model(nt)

sc = ex2.Spell_Checker()
# sc.build_model(nt, 3)
sc.add_language_model(lm)
print(sc.evaluate("popup"))
sc.add_error_tables(spelling_confusion_matrices.error_tables)
print(sc.spell_check('you should forget aboutit', alpha=0.95))

