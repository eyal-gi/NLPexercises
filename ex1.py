import re
import sys
import random
import math
import collections
from collections import defaultdict


class Ngram_Language_Model:
    """The class implements a Markov Language Model that learns a language model
        from a given text.
        It supports language generation and the evaluation of a given string.
        The class can be applied on both word level and character level.
    """

    def __init__(self, n=3, chars=False):
        """Initializing a language model object.
        Args:
            n (int): the length of the markov unit (the n of the n-gram). Defaults to 3.
            chars (bool): True iff the model consists of ngrams of characters rather then word tokens.
                          Defaults to False
        """
        self.n = n
        self.model_dict = defaultdict(int)  # a dictionary of the form {ngram:count}, holding counts of all ngrams in the specified text.
        self.chars = chars
        self.vocabulary = None   # a set of the types in the text
        self.nminus_dict = None

    def build_model(self, text):
        """populates the instance variable model_dict.

            Args:
                text (str): the text to construct the model from.
        """
        tokens = re.split(r'\s+', text)   # create a list of words out of the corpora
        tokens.remove('')

        # Every tuple of n words is joined to a string, and the Counter func creates a dict with counts
        self.model_dict = defaultdict(int, collections.Counter(" ".join(tuple(tokens[i:i+self.n])) for i in range(len(tokens)-self.n)))
        self.nminus_dict = collections.Counter(" ".join(tuple(tokens[i:i+self.n-1])) for i in range(len(tokens)-self.n-1))
        self.vocabulary = set(tokens)
        print(self.model_dict)   # TODO delete this line
        print(self.nminus_dict)   # TODO delete this line

        # TODO save all the dictionaries of <n for smoothing

    def get_model_dictionary(self):
        """Returns the dictionary class object
        """
        return self.model_dict

    def get_model_window_size(self):
        """Returning the size of the context window (the n in "n-gram")
        """
        return self.n

    def P(self, candidate):
        # context = sequence.rsplit(' ', 1)[0]   # remove the last word of the sentence
        context = candidate[0]
        sequence = candidate[2]
        print(self.model_dict[sequence]/self.nminus_dict[context])
        return self.model_dict[sequence]/self.nminus_dict[context]

    def candidates(self, context):
        """
        Returns a set of all possible ngrams sequences
        :param context:
        :return:
        """
        candi = set()   # initialize the candidates set
        for w in self.vocabulary:   # add every word in vocabulary to the given context
            c = context + " " + w   # add every word in vocabulary to the given context
            if c in self.model_dict:   # verify this sequence is in the model dictionary
                candi.add((context, w, c))

        #candi1 = set(context+" "+w for w in self.vocabulary if context+" "+w in self.model_dict)
        print(candi)
        #print(candi1)
        return candi

    def generate(self, context=None, n=20):
        """Returns a string of the specified length, generated by applying the language model
        to the specified seed context. If no context is specified the context should be sampled
        from the models' contexts distribution. Generation should stop before the n'th word if the
        contexts are exhausted. If the length of the specified context exceeds (or equal to)
        the specified n, the method should return the a prefix of length n of the specified context.

            Args:
                context (str): a seed context to start the generated string from. Defaults to None
                n (int): the length of the string to be generated.

            Return:
                String. The generated text.

        """
        context_len = len(re.split(r'\s+', context))
        if context==None:  #| len(context) < self.n-1:    # TODO change from len of context to no. of words
            # there is no given context or not big enough
            # add <s> or <s><s> or <s><s>...<s>
            str=None


        elif context_len > self.n:   # context is loner than self.n
            str=None

        else:   # context is same length as self.n
            str = context
            ngram = context
            for i in range(0, n-context_len):
                # chosen_candidate = max(self.candidates(ngram), key=self.P)   # return the string with highest probability
                candids = self.candidates(ngram)
                chosen = self.choose(candids)
                str = str + " " + chosen
                ngram = ' '.join(str.split()[len(str)-self.n:])
                print(ngram)
                # str = str + " " + chosen_candidate[1]   # adds the chosen word to the string
                # ngram = ' '.join(chosen_candidate[2].split()[1:])
            print(str)

        return str

    def choose(self, candidates):
        cands = {}
        print(candidates)
        for c in candidates:
            print(c)
            cands[c[1]] = self.P(c)
            print(cands)
            print(cands.values())
        mv = max(cands.values())
        print(mv)
        return random.choice([k for (k, v) in cands.items() if v == mv])

        print(cands)

        return

    def evaluate(self, text):
        """Returns the log-likelihood of the specified text to be a product of the model.
           Laplace smoothing should be applied if necessary.

           Args:
               text (str): Text to evaluate.

           Returns:
               Float. The float should reflect the (log) probability.
        """

    def smooth(self, ngram):
        """Returns the smoothed (Laplace) probability of the specified ngram.

            Args:
                ngram (str): the ngram to have it's probability smoothed

            Returns:
                float. The smoothed probability.
        """


def normalize_text(text):
    """Returns a normalized version of the specified string.
      You can add default parameters as you like (they should have default values!)
      You should explain your decisions in the header of the function.

      Args:
        text (str): the text to normalize

      Returns:
        string. the normalized text.
    """
    # TODO add a case when the model is for chars and not words
    nt = text.lower()   # lower-case the text
    # nt = re.sub('([.,!?()%^$&-])', r' \1 ', nt)   # add space before/after a punctuation
    # nt = re.sub(r'\s+', ' ', nt)   # remove unwanted spaces (more than 1)
    nt = re.sub('(?<! )(?=[.,:?!@#$%^&*()\[\]\\\])|(?<=[.,:?!@#$%^&*()\[\]\\\])(?! )', r' ', nt)
    return nt


def who_am_i():  # this is not a class method
    """Returns a dictionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    return {'name': 'Eyal Ginosar', 'id': '307830901', 'email': 'eyalgi@post.bgu.ac.il'}