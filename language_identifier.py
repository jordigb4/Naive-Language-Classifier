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
            if c > 5: # Elimineu tots el trigrams que apareguin menys de 5 vegades en el corpus
                tr_c[tr] = c
            count += c
        return tr_c, count
        
    
    def identify_language(self, path):

        preprocessed_test = self.__read_corpora(path)
        phrases = preprocessed_test.split("  ")

        for phrase in phrases:
            pred = max([self.likelihood(phrase, language) for language in self.languages])

    def likelihood(self, d, language):
        #Pre: d is a string/phrase
        #Return: probability of document associated to d writen in 'language'

        finder = TrigramCollocationFinder.from_words(d)
        sum_logprobs = sum([c*log(self.LID_n_gram_likelihood(tr, language)) for tr, c in finder.ngram_fd.items()])
        return exp(sum_logprobs)
 
    def LID_n_gram_likelihood(self, tr, language):
        #Pre: tr is a tri-gram character; language is recognizable by the model
        #Return: MLE with LID smoothing of tr belong to 'language' based on training corpora

        count_tr = self.get_count(tr, language)
        total_count = self.get_total(language)
        return (count_tr + 0.5) / (total_count + 0.5*pow(24, 3))

    def get_count(self, tr, language):
        #Pre: tr is a tri-gram character; language is recognizable by the model
        #Return: count of tr in training corpora related to language
        
        corpora = self.trigrams_corpora[language]
        return corpora[tr]

    