from Preprocessing import preprocessing
from nltk.collocations import TrigramCollocationFinder
import numpy as np

class LanguageIdentifier:

    def __init__(self,smoothing = "Lidstone", alpha = 0.5):

     
        self.smoothing = smoothing
        if smoothing == "Lidstone":
            self.alpha = alpha

        self.languages = ["deu", "eng", "fra", "ita", "nld", "por", "spa"]
        self.languages_encoding = {"deu" : 0, "eng" : 1, "fra" : 2, "ita" : 3, "nld" : 4, "por" : 5, "spa" : 6}

        self.__train()

    def __train(self):
        
        preprocessed_corpora = {language : self.__read_corpora(f"corpora/{language}_trn.txt") for language in self.languages}

        self.trigrams_corpora = {language : self.__get_trigrams(preprocessed_corpora[language]) for language in self.languages}
    
    def __read_corpora(self,path):

        with open(path, 'r', encoding='utf-8') as file:
            return preprocessing(file.read())
            
    def __get_trigrams(self,corpora):

        finder = TrigramCollocationFinder.from_words(corpora)
        return [tr for tr in finder.ngram_fd.items()]
    
    def identify_language(self,path):
        actual = np.array()
        predicted = np.array()
        correct = 0

        preprocessed_test = self.__read_corpora(path)
        phrases = preprocessed_test.split("  ")

