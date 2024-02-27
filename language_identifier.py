from Preprocessing import preprocessing
from nltk.collocations import TrigramCollocationFinder
import numpy as np
from math import log, exp

class LanguageIdentifier:

    def __init__(self, smoothing = "Lidstone", alpha = 0.5):
     
        self.smoothing = smoothing
        if smoothing == "Lidstone":
            self.alpha = alpha

        self.languages = ["deu", "eng", "fra", "ita", "nld", "por", "spa"]
        self.languages_encoding = {"deu" : 0, "eng" : 1, "fra" : 2, "ita" : 3, "nld" : 4, "por" : 5, "spa" : 6}

        self.__train()

    def __train(self):
        #Post: creates two attributes that keep training data features, i.e. the number of
        #      total trigrams in each language and, for each language the count of each trigram.
        
        self.total_trigrams_corpora = {} #total trigrams counts for each language
        self.trigrams_corpora = {} #trigrams counts for a language
        for language in self.languages:
            corpora = self.__read_corpora(f"corpora/{language}_trn.txt")
            self.trigrams_corpora[language], self.total_trigrams_corpora[language] = self.__get_trigrams(corpora)
    
    def __read_corpora(self, path):
        #Return: preprocessed string from file in path

        with open(path, 'r', encoding='utf-8') as file:
            return preprocessing(file.read())
            
    def __get_trigrams(self, corpora):
        #Pre: corpora is a string
        #Return: trigrams freq's from corpora and total count

        finder = TrigramCollocationFinder.from_words(corpora)
        tr_c = {}
        count = 0
        for tr, c in finder.ngram_fd.items():
            tr_c[tr] = c
            count += c
        return tr_c, count
        
    
    def identify_language(self, path):
        predicted = np.array()

        preprocessed_test = self.__read_corpora(path)
        phrases = preprocessed_test.split("  ")

        for phrase in phrases:

            finder = TrigramCollocationFinder.from_words(phrase)
            trigrams = [tr for tr in finder.ngram_fd.items()]

    def likelihood(self, d, language):
        #Pre: d is a sequence of character tri-grams
        #Return: probability of document associated to d writen in 'language'

        sum_logprobs = sum([log(self.LID_n_gram_likelihood(e_j, language)) for e_j in d])
        return exp(sum_logprobs)

    def LID_n_gram_likelihood(self, e_j, language):
        #Pre: e_j is a tri-gram character; language is recognizable by the model
        #Return: MLE with LID smoothing of e_j belong to 'language' based on training corpora

        count_e_j = self.get_count(e_j, language)
        total_count = self.get_total(language)
        return (count_e_j + 0.5) / (total_count + 0.5*pow(24, 3))

    def get_count(self, e_j, language):
        #Pre: e_j is a tri-gram character; language is recognizable by the model
        #Return: count of e_j in training corpora related to language
        corpora = self.trigrams_corpora[language]
        return corpora[e_j]

    