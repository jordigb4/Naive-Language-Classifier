from Preprocessing.preprocessing import preprocessing
from nltk.collocations import TrigramCollocationFinder
import numpy as np
from math import log, exp

class LanguageIdentifier:

    def __init__(self, smoothing = "Lidstone", alpha = 0.5)-> None:
     
        self.smoothing = smoothing
        if smoothing == "Lidstone":
            self.alpha = alpha

        self.languages = ["deu", "eng", "fra", "ita", "nld", "spa"]
        self.languages_encoding = {"deu" : 0, "eng" : 1, "fra" : 2, "ita" : 3, "nld" : 4, "spa" : 5}

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
        ct = 0
        for tr, c in finder.ngram_fd.items():
            if c > 5: # Elimineu tots el trigrams que apareguin menys de 5 vegades en el corpus
                tr_c[tr] = c
            ct += c
        return tr_c, ct
        
    def identify_language(self, path):

        preprocessed_test = self.__read_corpora(path)
        phrases = preprocessed_test.split("  ")

        predicted = []
        for phrase in phrases:
            phrase_probs = [(language, self.likelihood(phrase, language)) for language in self.languages]
            pred = max(phrase_probs, key = lambda x: x[1])[0]
            predicted.append(self.languages_encoding[pred])

        return np.array(predicted)

    def likelihood(self, d, language):
        #Pre: d is a string/phrase
        #Return: probability of document associated to d writen in 'language'

        finder = TrigramCollocationFinder.from_words(d)
        sum_logprobs = sum([c*log(self.LID_n_gram_likelihood(tr, language)) for tr, c in finder.ngram_fd.items()])
        return sum_logprobs
 
    def LID_n_gram_likelihood(self, tr, language):
        #Pre: tr is a tri-gram character; language is recognizable by the model
        #Return: MLE with LID smoothing of tr belong to 'language' based on training corpora

        ct_tr = self.get_count(tr, language)
        total_ct = self.total_trigrams_corpora[language] 
        return (ct_tr + self.alpha) / (total_ct + self.alpha*pow(24, 3))

    def get_count(self, tr, language):
        #Pre: tr is a tri-gram character; language is recognizable by the model
        #Return: count of tr in training corpora related to language
        
        corpora = self.trigrams_corpora[language]
        ct = corpora.get(tr, 0) #if not found count is 0
        return ct
    