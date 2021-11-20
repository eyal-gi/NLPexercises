import re
import sys
import random
import math
import collections
from collections import defaultdict
import nltk


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
        self.model_dict = defaultdict(
            int)  # a dictionary of the form {ngram:count}, holding counts of all ngrams in the specified text.
        self.chars = chars
        self.__vocabulary = None  # a set of the types in the text
        self.ngrams_dict = defaultdict(int)  # dictionary of ngrams dictionaries

    def build_model(self, text):
        """populates the instance variable model_dict.

            Args:
                text (str): the text to construct the model from.
        """

        if self.chars is True:
            tokens = list(text)
            self.model_dict = defaultdict(int, collections.Counter(
                text[i:i + self.n] for i in range(len(tokens) - self.n + 1)))
            for j in range(self.n):
                self.ngrams_dict[self.n - j] = defaultdict(int, collections.Counter(
                    text[i:i + self.n - j] for i in range(len(tokens) - (self.n - j))))

        else:
            tokens = re.split(r'\s+', text)  # create a list of words out of the corpora
            # Every tuple of n words is joined to a string, and the Counter func creates a dict with counts
            self.model_dict = defaultdict(int, collections.Counter(
                " ".join(tuple(tokens[i:i + self.n])) for i in range(len(tokens) - self.n + 1)))
            # a dictionary of every possible n-gram dictionary. This is used for evaluation and generation with context smaller then n
            for j in range(self.n):
                self.ngrams_dict[self.n - j] = defaultdict(int, collections.Counter(
                    " ".join(tuple(tokens[i:i + self.n - j])) for i in range(len(tokens) - (self.n - j))))

        self.__vocabulary = set(tokens)

    def get_model_dictionary(self):
        """Returns the dictionary class object
        """
        return self.model_dict

    def get_model_window_size(self):
        """Returning the size of the context window (the n in "n-gram")
        """
        return self.n

    def P(self, candidate, context, given_n=None):
        """Returns the probability of a given candidate word to follow the given context.
            By default, the functions calculates according to the model's n.
            In case a lower-n is needed, it can be provided and the function will calculate accordingly.

            Args:
                candidate(str): the candidate word to follow the context
                context(str): the context to follow
                given_n(int): if needed, a different n for the ngram

            Return:
                Float. The probability of the word for the context.

        """
        sequence = context.copy()  # the sequence begin with the context
        sequence.append(candidate)  # append the candidate word with the context

        # calculate according to the normal ngram algorithm
        if given_n is None:
            if self.chars is not True:
                return self.model_dict[" ".join(sequence)] / self.ngrams_dict[self.n - 1][" ".join(context)]
            else:
                return self.model_dict["".join(sequence)] / self.ngrams_dict[self.n - 1]["".join(context)]

        # calculate based on a different, given n.
        if self.chars is not True:
            return self.ngrams_dict[given_n][" ".join(sequence)] / self.ngrams_dict[given_n - 1][" ".join(context)]
        else:
            return self.ngrams_dict[given_n]["".join(sequence)] / self.ngrams_dict[given_n - 1]["".join(context)]

    def p_first(self, word):
        """Returns the probability for a given word to be the first in a context.
        As padding has not been implemented in this model, a calculation based on a word to appear first in a sentence
        is made.

            Args:
                word(str): the word to calculate its probability

            Return:
                Float. probability to be first
         """

        # if n is equal to 1, the probability is the word count above all words count.
        if self.n == 1:
            return self.model_dict[word] / sum(self.model_dict.values())

        # the probability is the word count above all possible ngrams count.
        else:
            return self.ngrams_dict[1][word] / sum(self.model_dict.values())

    def candidates(self, context, n_gram=None):
        """Returns a set of all possible ngrams sequences

            Args:
                context (list): the context to create candidates from
                n_gram(int): if generating for n_gram < self.n, generates for the relevant dictionary

            Return:
                List. The candidates words.
        """
        candi = set()  # initialize the candidates set

        for w in self.__vocabulary:  # add every word in vocabulary to the given context
            c = context.copy()
            c.append(w)

            if n_gram is None:  # regular ngram
                if self.chars is not True:
                    if " ".join(c) in self.model_dict: candi.add(w)  # verify this sequence is in the model dictionary
                else:
                    if "".join(c) in self.model_dict: candi.add(w)
            else:  # different n for ngram.
                if self.chars is not True:
                    if " ".join(c) in self.ngrams_dict[n_gram]: candi.add(w)
                else:
                    if "".join(c) in self.ngrams_dict[n_gram]: candi.add(w)

        # candi1 = set(context+" "+w for w in self.__vocabulary if context+" "+w in self.model_dict)
        return candi

    def generate_unigram(self, context, n):
        """Returns a string of the specified length based on words probabilities from the model's dictionary

            Args:
                context(string): The given context. Only used for the total word count, as the generation is not
                                    context based.
                n(int): The string length.

            Return:
                String. The generated text.

        """
        str = context
        for i in range(n - len(context)):
            # Randomly choose a word for the words distribution.
            str.append(
                random.choices(population=list(self.model_dict.keys()), weights=self.model_dict.values(), k=1)[0])

        if self.chars is not True:
            return " ".join(str)
        else:
            return ''.join(str)

    def generate(self, context=None, n=20):
        """Returns a string of the specified length, generated by applying the language model
        to the specified seed context. If no context is specified the context should be sampled
        from the models' contexts distribution. Generation should stop before the n'th word if the
        contexts are exhausted. If the length of the specified context exceeds (or equal to)
        the specified n, the method should return the a prefix of length n of the specified context.

        If context length is lower than self.n, a stupid backoff smoothing is applied to determine the next word.

            Args:
                context (str): a seed context to start the generated string from. Defaults to None
                n (int): the length of the string to be generated.

            Return:
                String. The generated text.

        """

        if context == None: context = ''  # no context was given.
        if len(context) == 0:
            if self.n == 1:  # if this is a unigram model, generate from the unigram function
                empty = ['']
                return self.generate_unigram(empty, n)
            # choose a starting context randomly based on the context's distribution
            context = random.choices(population=list(self.ngrams_dict[self.n - 1].keys()),
                                     weights=self.ngrams_dict[self.n - 1].values())[0]
        if self.chars is True:
            context_l = list(context)
        else:
            context_l = re.split(r'\s+', context)  # the context as a list of words

        str = context_l.copy()  # the generates string start for the context

        # if the context is longer than n, returns a subset string.
        if len(context_l) >= n:
            if self.chars is not True:
                return " ".join(context_l[0:n])
            else:
                return ''.join(context_l[0:n])

        # if this is a unigram model, generate from the unigram function
        if self.n == 1: return self.generate_unigram(context_l, n)

        # context is shorter than the needed length
        if len(context_l) < self.n - 1:
            ngram = context_l.copy()
            for current_n in range(len(context_l) + 1, min(n, self.n)):  # generate enough words for context for ngram
                cands = self.candidates(ngram, current_n)
                if not cands:
                    if self.chars is not True:
                        return " ".join(str)  # no possible candidates -> end function.
                    else:
                        return ''.join(str)
                chosen = self.choose(cands, ngram,
                                     current_n)  # choose the word with highest probability from the list of options for next word
                str.append(chosen)
                ngram.append(chosen)


        # context is longer than needed
        elif len(context_l) > self.n - 1:
            ngram = (context_l[len(context_l) - (self.n - 1):]).copy()

        # context is exactly self.n - 1
        else:
            ngram = context_l.copy()

        for i in range(0, n - len(context_l)):
            cands = self.candidates(ngram.copy())
            if not cands: break  # no possible candidates -> end function.
            chosen = self.choose(cands,
                                 ngram.copy())  # choose the word with highest probability from the list of options for next word
            str.append(chosen)
            ngram.append(chosen)
            ngram.pop(0)

        if self.chars is not True:
            return " ".join(str)
        else:
            return ''.join(str)

    def choose(self, candidates, context_, n_gram=None):
        """Return the word with the highest probability to be next in the sentence based on ngrams.
        If there are more than one word with the name probability, a random choice is made.

        Args:
            candidates (list): list of possible candidates, based on ngram algorithm

        Return:
            The chosen word (str)
        """
        context = context_.copy()

        probs = {}  # dictionary of the candidates with their probabilities
        for c in candidates:  # c is a tuple of (ngram list, predicted word)
            probs[c] = self.P(c, context, n_gram)  # calculate the probability for each candidate.

        return (random.choices(population=list(probs.keys()), weights=probs.values(), k=1))[0]

    def evaluate(self, text):
        """Returns the log-likelihood of the specified text to be a product of the model.
           Laplace smoothing should be applied if necessary.

           Words that are OOV (out of vocabulary) or context that doesn't appear in the model dictionary are treated by
           laplace smoothing. If laplace smoothing has been applied, all further words will be smoothed.
           The first n-1 words are evaluated by stupid backoff smoothing.

           Args:
               text (str): Text to evaluate.

           Returns:
               Float. The float should reflect the (log) probability.
        """
        apply_smoothing = False  # states is smoothing has been applied

        if not text:  # no text to evaluate
            raise Exception('A test must be inserted for evaluation')
        text = normalize_text(text)
        log_probs = []

        if self.chars is not True:
            text_list = re.split(r'\s+', text)  # split the text to a list of words
        else:
            text_list = list(text)

        text_len = len(text_list)  # text word count

        # Unigram model
        if self.n == 1:
            for w in text_list:
                if w in self.__vocabulary and apply_smoothing == False:
                    log_probs.append(math.log(self.model_dict[w] / sum(self.model_dict.values())))
                else:  # word is OOV
                    apply_smoothing = True
                    log_probs.append(math.log(self.smooth(w, 1)))
            return round(sum(log_probs), 3)

        # Evaluation for the first word:
        if text_list[0] in self.__vocabulary:  # word from vocabulary
            log_probs.append(
                math.log(self.p_first(text_list[0])))  # calc first word based on the model's context distribution
        else:  # word OOV
            log_probs.append(math.log(self.smooth(text_list[0], 1)))
            apply_smoothing = True

        for n in range(2, min(text_len,
                              self.n)):  # calc probability for words until there are enough for ngram according to self.n
            if self.chars is not True:
                ngram = " ".join(text_list[0:n])
                nm_gram = " ".join(text_list[0:n - 1])
            else:
                ngram = "".join(text_list[0:n])
                nm_gram = "".join(text_list[0:n - 1])

            if ngram in self.ngrams_dict[n] and apply_smoothing == False:  # word from vocab and no smoothing
                log_probs.append(math.log(self.ngrams_dict[n][ngram] / self.ngrams_dict[n - 1][nm_gram]))
            else:  # word is OOV or smoothing was applied
                apply_smoothing = True
                log_probs.append(math.log(self.smooth(ngram, nm_gram, n)))

        if text_len >= self.n:
            for i in range(0, text_len - (self.n - 1)):
                if self.chars is not True:
                    ngram = " ".join(text_list[i:i + self.n])
                    nm_gram = " ".join(text_list[i:i + self.n - 1])
                else:
                    ngram = "".join(text_list[i:i + self.n])
                    nm_gram = "".join(text_list[i:i + self.n - 1])

                if ngram in self.model_dict and apply_smoothing == False:
                    log_probs.append(math.log(self.model_dict[ngram] / self.ngrams_dict[self.n - 1][nm_gram]))
                else:
                    apply_smoothing = True
                    log_probs.append(math.log(self.smooth(ngram, nm_gram)))

        return round(sum(log_probs), 3)

    def smooth(self, ngram, nm_gram=None, given_n=None):
        """Returns the smoothed (Laplace) probability of the specified ngram.

            Args:
                ngram (str): the ngram to have it's probability smoothed

            Returns:
                float. The smoothed probability.
        """
        # Initialize count values
        c_ngram = 0
        c_context = 0

        # default model
        if given_n is None:
            if ngram in self.model_dict:  # ngram from vocabulary
                c_ngram = self.model_dict[ngram]
            if nm_gram in self.ngrams_dict[self.n - 1]:  # n-1 gram from vocabulary
                c_context = self.ngrams_dict[self.n - 1][nm_gram]
        # not default model
        else:
            if ngram in self.ngrams_dict[given_n]:  # ngram from vocabulary
                c_ngram = self.ngrams_dict[given_n][ngram]
            if nm_gram in self.ngrams_dict[given_n - 1]:  # n-1 gram from vocabulary
                c_context = self.ngrams_dict[given_n - 1][nm_gram]

        if given_n is None:
            v = len(self.ngrams_dict[self.n - 1])  # default model
        else:
            v = len(self.ngrams_dict[given_n - 1])  # not default model

        # Unigram model
        if given_n is not None and given_n == 1:
            return (c_ngram + 1) / (len(self.__vocabulary) + 1)

        return (c_ngram + 1) / (c_context + v)


def normalize_text(text):
    """Returns a normalized version of the specified string.
      You can add default parameters as you like (they should have default values!)
      You should explain your decisions in the header of the function.

      Args:
        text (str): the text to normalize

      Returns:
        string. the normalized text.
    """
    # todo: add- remove ' from text
    nt = text.lower()  # lower-case the text

    nt = re.sub('(?<! )(?=[._=,\-:?\"!@#$%^&*()\[\]\\\])|(?<=[._=,\-:?\"!@#$%^&*()\[\]\\\])(?! )', r' ', nt)
    text = text.replace('\n', " ")
    text = text.replace('...', " ")
    text = text.replace('#', " ")
    text = text.replace('\\', " ")
    text = text.replace('@', " ")
    text = text.replace("---", " ")
    text = text.replace("--", " ")
    text = text.replace("-", " ")
    text = text.replace('_', " ")
    text = text.replace('"', " ")
    text = text.replace('\'', "")
    text = text.replace(':', "")
    text = text.replace('=', "")
    text = text.replace('!', "")
    text = text.replace('*', "")

    tokens = re.split(r'\s+', nt)  # create a list of words out of the corpora
    if tokens[-1] == '':
        tokens.pop()

    nt = " ".join(tokens)
    return nt


class Spell_Checker:
    """The class implements a context sensitive spell checker. The corrections
        are done in the Noisy Channel framework, based on a language model and
        an error distribution model.
    """

    def __init__(self, lm=None):
        """Initializing a spell checker object with a language model as an
        instance  variable. The language model should support the evaluate()
        and the get_model() functions as defined in assignment #1.

        Args:
            lm: a language model object. Defaults to None
        """

        self.lm = lm
        self.error_tables = {}
        self.__vocabulary = set()
        self.__chars_dict = {}
        self.__alpha = None
        self.unigrams = {}

    def build_model(self, text, n=3):
        """Returns a language model object built on the specified text. The language
            model should support evaluate() and the get_model() functions as defined
            in assignment #1.

            Args:
                text (str): the text to construct the model from.
                n (int): the order of the n-gram model (defaults to 3).

            Returns:
                A language model object
        """

        self.lm = Ngram_Language_Model(n=n)
        self.lm.build_model(text=text)
        self.__vocabulary = set(re.split(r'\s+', normalize_text(text)))  # create a list of words out of the corpora)
        self.__chars_dict = self._create_chars_dict(text)

    def add_language_model(self, lm):
        """ Adds the specified language model as an instance variable.
            (Replaces an older LM dictionary if set)
            Also creates a new vocabulary and chars dictionary based on the language model ngrams dictionary

            Args:
                lm: a language model object
        """
        self.lm = lm
        self._create_vocabulary()
        self.__chars_dict = self._create_chars_dict()

    def learn_error_tables(self, errors_file):
        """ Returns a nested dictionary {str:dict} where str is in:
            <'deletion', 'insertion', 'transposition', 'substitution'> and the
            inner dict {str: int} represents the confusion matrix of the
            specific errors, where str is a string of two characters mattching the
            row and culumn "indixes" in the relevant confusion matrix and the int is the
            observed count of such an error (computed from the specified errors file).
            Examples of such string are 'xy', for deletion of a 'y'
            after an 'x', insertion of a 'y' after an 'x'  and substitution
            of 'x' (incorrect) by a 'y'; and example of a transposition is 'xy' indicates the characters that are transposed.


            Notes:
                1. Ultimately, one can use only 'deletion' and 'insertion' and have
                    'substitution' and 'transposition' derived. Again,  we use all
                    four types explicitly in order to keep things simple.
            Args:
                errors_file (str): full path to the errors file. File format, TSV:
                                    <error>    <correct>


            Returns:
                A dictionary of confusion "matrices" by error type (dict).
        """

        errors = []
        with open(errors_file) as errors_text:
            for line in errors_text:
                line = line.strip().split('\t')

                # add the erroneous word and corrected word to errors list as a tuple of (x, w)
                errors.append((line[0].lower(), line[1].lower()))

        errors_table = {'deletion': {},
                        'insertion': {},
                        'transposition': {},
                        'substitution': {}
                        }

        for e in errors:
            e_type = self._identify_error(e[0], e[1])
            chars = self._find_chars(e[0], e[1], e_type)
            if chars not in errors_table[e_type].keys():
                errors_table[e_type][chars] = 1
            else:
                errors_table[e_type][chars] += 1

        self.error_tables = errors_table

    def _identify_error(self, x, w):
        """ Returns the type of error that is found between erroneous word x and correct word w

            Args:
                x (str): Erroneous word
                w (str): Correct word

            Return:
                Type of error (str): deletion / insertion / substitution / transposition
        """
        if len(x) < len(w):
            return 'deletion'

        elif len(x) > len(w):
            return 'insertion'

        elif sorted(x) == sorted(w):
            return 'transposition'

        else:
            return 'substitution'


    def _find_chars(self, x, w, error_type):
        """ Returns the 2 chars that represent the error.
            For deletion: return [wi-1, wi]
            For insertion: return [wi-1, xi]
            For substitution: return [xi, wi]
            For transposition: return [wi, wi+1]

            Args:
                x (str): Erroneous word
                w (str): Correct word
                error_type (str): deletion / insertion / substitution / transposition

            Return:
                chars (str): two chars.

        """
        if error_type == 'deletion':
            for i in range(len(x)):
                if x[i] != w[i]:
                    if i == 0:
                        return '#' + w[i]
                    else:
                        return w[i - 1] + w[i]

        elif error_type == 'insertion':
            for i in range(len(w)):
                if x[i] != w[i]:
                    if i == 0:
                        return '#' + x[i]
                    else:
                        return w[i - 1] + x[i]

        elif error_type == 'substitution':
            for i in range(len(w)):
                if x[i] != w[i]:
                    return x[i] + w[i]

        else:  # error_type == 'transposition'
            for i in range(len(w)):
                if x[i] != w[i]:
                    return w[i] + w[i + 1]


    def _create_vocabulary(self):
        """ Creates the vocabulary of the language model based on the model dictionary
            Both types vocabulary and unigram dictionary are generated.
        """
        tokens = set()
        unigram = {}

        # go over all the ngrams in the model and add the first word of the ngram to the tokens list.
        for ng in self.lm.get_model_dictionary().keys():
            words = re.split(r'\s+', ng)
            word = words[0]
            tokens.add(word)
            if word not in unigram.keys():
                unigram[word] = self.lm.get_model_dictionary()[ng]
            else:
                unigram[word] += self.lm.get_model_dictionary()[ng]

        # Since the I only take the first word of every ngram, for the last on I insert all the words in it.
        last_words = re.split(r'\s+', list(self.lm.get_model_dictionary().keys())[-1])
        for w in last_words[1:]:
            tokens.add(w)
            if w not in unigram.keys():
                unigram[w] = self.lm.get_model_dictionary()[ng]
            else:
                unigram[w] += self.lm.get_model_dictionary()[ng]

        self.__vocabulary = tokens
        self.unigrams = defaultdict(int, unigram)

    def _create_chars_dict(self, text=None):
        """ Create unigram and bigram chars dictionaries.
            If the language model is created from scratch, the dictionaries are calculated directly
            from the given text.
            If a language model is added, the dictionaries are calculated from the lm ngram dictionary.
        """
        # If a language model is added and not created:
        if text is None:
            dict = { 'bigram': { },
                     'unigram': { }
                     }
            for j in range(2):  # create two dictionaries: 2 chars and 1 char
                if j == 0: ngram = 'bigram'
                else: ngram = 'unigram'

                # go over each unigram from the language model
                for key in self.unigrams.keys():
                    # take 2 chars or 1 char from the word
                    for i in range( len(key) - (2-j) ):
                        chars = key[i:i + 2 - j]
                        # count the number of appearances based on the unigram dictionary
                        if chars not in dict[ngram].keys():
                            dict[ngram][chars] = self.unigrams[key]
                        else:   # add value
                            dict[ngram][chars] += self.unigrams[key]

        # -------------------------------------------------------------------- #

        # if language model is created:
        else:
            tokens = list(text)
            dict = {}
            for j in range(2):
                if j == 0: ngram = 'bigram'
                else: ngram = 'unigram'
                dict[ngram] = defaultdict(int, collections.Counter(
                    text[i:i + 2 - j] for i in range(len(tokens) - (2 - j))))
        return dict

    def add_error_tables(self, error_tables):
        """ Adds the speficied dictionary of error tables as an instance variable.
            (Replaces an older value disctionary if set)

            Args:
                error_tables (dict): a dictionary of error tables in the format
                returned by  learn_error_tables()
        """
        self.error_tables = error_tables

    def evaluate(self, text):
        """Returns the log-likelihood of the specified text given the language
            model in use. Smoothing is applied on texts containing OOV words

           Args:
               text (str): Text to evaluate.

           Returns:
               Float. The float should reflect the (log) probability.
        """
        return self.lm.evaluate(text=text)

    def spell_check(self, text, alpha):
        """ Returns the most probable fix for the specified text. Use a simple
            noisy channel model if the number of tokens in the specified text is
            smaller than the length (n) of the language model.

            Args:
                text (str): the text to spell check.
                alpha (float): the probability of keeping a lexical word as is.

            Return:
                A modified string (or a copy of the original if no corrections are made.)
        """

        if text == '':
            return ''
        self.__alpha = alpha
        candidates_dict = self._candidates(normalize_text(text))  # a dictionary of a candidate and it's score.
        print(candidates_dict)   # todo: delete this line


        return max(candidates_dict.keys(), key=candidates_dict.get)  # return the string with the highest score

    def _candidates(self, text):
        """ Returns a set of possible corrected texts. Each word in the text will get a set of candidates
            of its own, resulting in a set of all possible texts when only one word is corrected in each text.
            The text itself is also considered a valid candidate.

            Args:
                text (str): text to create candidates from.

            Return:
                A dictionary of candidates text and the probability of the correction - P(x|w)P(W)

        """
        candidates = {}
        text_list = re.split(r'\s+', text)  # the text as list of words
        check_words = [t for t in text_list if t not in self.__vocabulary]   # initially only look for OOV words.

        # if all words are from vocabulary, check all words
        if not check_words:
            check_words = text_list.copy()

        for w in check_words:
            word_candidates = self._word_candidates(w)  # get all possible candidates and their probability for word w
            for c in word_candidates:
                new_text = text_list.copy()
                self._list_replace(lst=new_text, old=w, new=c)  # create the new text with corrected word
                corrected_candidate = ' '.join(new_text)    # join the word list to a single string
                word_pxw = word_candidates[c]   # P(x|w) of the candidate c
                text_eval = self.evaluate(corrected_candidate)   # P(W) of the whole corrected candidate

                # add the corrected candidate with its score to the candidates dictionary
                candidates[corrected_candidate] = math.log(word_pxw) + text_eval

        return candidates

    def _word_candidates(self, word):
        """ Returns a dictionary of all possible corrections to the given word. The returned words will only
            be known words from the language model corpora. If the given word nor any of its edits is a
            vocabulary word, the function will return the given word with a default smoothed probability.
            The corrections are under the assumption that there is only 1 error in the word (i.e., deletion/
            insertion/substitution/transposition).

            Args:
                word(str): word to create candidates from.

            Return:
                A dictionary of candidates words and their probabilities. 'word':, 'pxw':
         """
        word_candidates = {}
        word_candidates_list = self._edits1(word)  # all the possible edits of word
        # add the original word as a candidate as well:
        word_candidates_list.append({'word': word, 'chars': '', 'error': 'original'})
        word_candidates_list = self._known(word_candidates_list)  # subset only known words

        # if the original word nor any edit is a word from the vocabulary,
        # return only the original word with a smoothed probability:
        if not word_candidates_list:
            word_candidates[word] = 1 / sum(self.__chars_dict['bigram'].values())
            return word_candidates

        # calculate probability P(x|w):
        for c in word_candidates_list:
            word_candidates[c['word']] = self._Pxw(c)

        return word_candidates

    def _known(self, words):
        """The subset of `words` that appear in the language model vocabulary."""
        known = []
        for w in words:
            if w['word'] in self.__vocabulary and w not in known:  # note: w['word'] is the corrected word
                known.append(w)

        return known

    def _edits1(self, word):
        """ All edits that are one edit away from `word`.
            For every edit that is made, a dictionary of {word, chars, error} is created, where:
            word - the corrected word
            chars - the chars where the error was made
            error - type of error (deletion/insertion/substitution/transposition)

            Args:
                word(str) - erroneous word to correct

            Return:
                dictionary - A list of dictionaries.
        """

        letters = 'abcdefghijklmnopqrstuvwxyz\''
        edits = []

        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

        for L, R in splits:
            for c in letters:

                # deletion errors
                if L:
                    chars = L[-1] + c  # chars=[wi-1,wi]  (wi-1=L[-1], wi=c)
                else:
                    chars = '#' + c
                edits.append({'word': L + c + R, 'chars': chars, 'error': 'deletion'})

                # substitution errors
                if R:
                    chars = R[0] + c  # chars=[xi,wi] (xi=R[0], wi=c)
                    edits.append({'word': L + c + R[1:], 'chars': chars, 'error': 'substitution'})

            # insertion errors
            if R:
                # chars is the inserted char and the char before it.
                if L:
                    chars = L[-1] + R[0]  # chars=[wi-1,xi] (wi-1=L[-1], xi=R[0])
                else:
                    chars = '#' + R[0]
                edits.append({'word': L + R[1:], 'chars': chars, 'error': 'insertion'})

            # transposition errors
            if len(R) > 1:
                chars = R[1] + R[0]   # chars=[wi, wi+1]
                edits.append({'word': L + R[1] + R[0] + R[2:], 'chars': chars, 'error': 'transposition'})

        return edits

    def _Pxw(self, candidate: dict) -> float:
        """ Returns the conditional probability of the x given w

            Args:
                candidate - A dict of (word, chars, error)

            Return:
                The probability on x given w.
        """
        chars = candidate['chars']
        error_type = candidate['error']
        # initiating variables:
        error_count = 0
        normalization = 0

        if error_type == 'original':
            return self.__alpha

        if error_type == 'deletion':
            # calculate counter:
            if chars in self.error_tables['deletion']:
                error_count = self.error_tables['deletion'][chars]

            # calculate denominator:
            if chars[0] == '#': chars = ' ' + chars[1]
            if chars in self.__chars_dict['bigram']:
                normalization = self.__chars_dict['bigram'][chars]

        elif error_type == 'insertion':
            # calculate counter:
            if chars in self.error_tables['insertion']:
                error_count = self.error_tables['insertion'][chars]
            # calculate denominator:
            if chars[0] == '#': chars = ' '
            if chars[0] in self.__chars_dict['unigram']:
                normalization = self.__chars_dict['unigram'][chars[0]]

        elif error_type == 'substitution':
            # calculate counter:
            if chars in self.error_tables['substitution']:
                error_count = self.error_tables['substitution'][chars]
            # calculate denominator:
            if chars[1] in self.__chars_dict['unigram']:
                normalization = self.__chars_dict['unigram'][chars[1]]

        else:   # error_type is 'transposition'
            # calculate counter:
            if chars in self.error_tables['transposition']:
                error_count = self.error_tables['transposition'][chars]
            # calculate denominator:
            if chars in self.__chars_dict['bigram']:
                normalization = self.__chars_dict['bigram'][chars]

        if error_count == 0:
            error_count = 1
        if normalization == 0:
            normalization = sum(self.__chars_dict['bigram'].values())

        return (error_count / normalization) * (1-self.__alpha)

    def _list_replace(self, lst, old, new):
        """replace list elements (inplace)"""
        i = -1
        try:
            while 1:
                i = lst.index(old, i + 1)
                lst[i] = new
        except ValueError:
            pass


def who_am_i():  # this is not a class method
    """Returns a ductionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    return {'name': 'Eyal Ginosar', 'id': '307830901', 'email': 'eyalgi@post.bgu.ac.il'}
