from Preprocessing.preprocessing import preprocessing
from nltk.collocations import TrigramCollocationFinder
import numpy as np
from math import log, exp
import warnings

class LanguageIdentifier:

    def __init__(self, smoothing = "Lidstone", alpha = 0.5, delta = None)-> None:
        """
            Initialize LanguageIdentifier with provided arguments.

            Parameters:

            smooting (str): Smoothing method. It must be one of the following:
                - Lidstone: Lidstone smoothing.
                - Abs_disc: Absolute discounting smoothing.

            alpha (float): Alpha/Lambda parameter value for Lidstone smoothing. Must be in the range (0,1)
                           default = 0.5 according to literature.

            delta (float): Delta parameter value for Absolute Discounting smoothing. Must be in range(0,1).
                           default is computed according to:
                                Ney, H., U. Essen, and R. Kneser. 1994. On structuring probabilistic dependencies in stochastic language modelling.
                                Computer Speech and Language, 8:1-38.
        """
        assert smoothing == 'Lidstone' or smoothing == 'Abs_disc', "Not Available Smoothing"
        assert 0 < alpha < 1, "Alpha must be in range (0, 1)" #alpha between 0 and 1
        assert delta is None or 0 < delta < 1, "Delta must be in range (0, 1)" #delta between 0 and 1

        self.smoothing = smoothing
        self.languages = ["deu", "eng", "fra", "ita", "nld", "spa"]  #supported languages

        self.__train()

        if smoothing == "Lidstone":
            self.alpha = alpha

        if smoothing == "Abs_disc":
            self.delta = delta if delta else (self.n1 / (self.n1 + 2 * self.n2))

    def __train(self):
        """
        Method to train the language identifier with the corpora associated to this file.

        Post-conditions: Creates two attributes that keep training data features, i.e. the number of
              total trigrams in each language and, for each language the count of each trigram (int,int).
        """

        
        self.total_trigrams_corpora = {} #total trigrams counts for each language
        self.trigrams_corpora = {} #trigrams counts for a language
        for language in self.languages:
            corpora = self.__read_corpora(f"corpora/{language}_trn.txt")
            self.trigrams_corpora[language], self.total_trigrams_corpora[language] = self.__get_trigrams(corpora)
    
    def __read_corpora(self, path):
        """
        Method to read and preprocess the corpora of a given path.

        Pre-conditions: path is a valid path to a readable file.

        Post-conditions: Returns the preprocessed file as raw text (str).
        """

        with open(path, 'r', encoding='utf-8') as file:

            return preprocessing(file.read())
            
    def __get_trigrams(self, corpora):
        """
        Method to get trigram's total count.

        Pre-conditions: corpora is a preprocessed string.

        Post-conditions: Returns trigrams freq's from corpora and total count (int,int).
        """


        finder = TrigramCollocationFinder.from_words(corpora)
        tr_c = {}
        ct = 0
        self.n1, self.n2 = 0, 0
        for tr, c in finder.ngram_fd.items():
            if c == 1: self.n1 += 1
            if c == 2: self.n2 += 1
            if c > 5: # Elimineu tots el trigrams que apareguin menys de 5 vegades en el corpus
                tr_c[tr] = c
            ct += c
        return tr_c, ct
        
    def identify_language(self, path):
        """
        Method to predict the language/s of given sentences.

        Pre-conditions: Path is a string that represents either a path or a sentence

        Post-conditions: Returns vector of predictions associated to the sentences on the path file, or to the string associated. 
                        Predictions are in the ISO3 format associated to the language (str).
        """
        
        try:
            preprocessed_test = self.__read_corpora(path)
        except:
            warnings.warn("Path not specified or incorrect, interpreting as string.")
            preprocessed_test = preprocessing(path)

        phrases = preprocessed_test.split("  ")
        predicted = []
        for phrase in phrases:
            phrase_probs = [(language, self.__likelihood(phrase, language)) for language in self.languages]
            pred = max(phrase_probs, key = lambda x: x[1])[0]
            predicted.append(pred)

        return np.array(predicted)

    def __likelihood(self, d, language):
        """
        Compute the probability of string associated to language.

        Pre-conditions: d is a string/phrase, language is a language from languages

        Post-conditions: Returns probability of document associated to d writen in 'language' (float)
        """

        finder = TrigramCollocationFinder.from_words(d)
        if self.smoothing == "Lidstone":
            sum_logprobs = sum([c*log(self.__LID_n_gram_likelihood(tr, language)) for tr, c in finder.ngram_fd.items()])
        if self.smoothing == "Abs_disc":
            sum_logprobs = sum([c*log(self.__ABS_n_gram_likelihood(tr, language)) for tr, c in finder.ngram_fd.items()])
        return sum_logprobs
 
    def __LID_n_gram_likelihood(self, tr, language):
        """
        Compute MLE with LID smoothing of trigram belonging to a certain language.

        Pre-conditions: tr is a tri-gram character; language is a language from languages.

        Post-conditions: Returns MLE with LID smoothing of tr belong to 'language' based on training corpora (float).
        """

        ct_tr = self.__get_count(tr, language)
        total_ct = self.total_trigrams_corpora[language] 
        return (ct_tr + self.alpha) / (total_ct + self.alpha*pow(24, 3))
    
    def __ABS_n_gram_likelihood(self, tr, language):
        """
        Compute MLE with Abs_disc smoothing of trigram belonging to a certain language.

        Pre-conditions: tr is a tri-gram character; language is a language from languages.

        Post-conditions: Returns MLE with Abs_disc smoothing of tr belong to 'language' based on training corpora (float).
        """
        d = self.delta
        ct_tr = self.__get_count(tr, language)
        total_ct = self.total_trigrams_corpora[language] # nº of occurrences of trigrams in train
        n_0_counts = pow(24, 3) - len(self.trigrams_corpora[language]) # nº of trigrams not seen = potential - seen different

        if ct_tr > 0:
            return ((ct_tr - d) / total_ct)
        else:
            return ((pow(24, 3) - n_0_counts) * d / n_0_counts) / total_ct
        
    def __get_count(self, tr, language):
        """
        Method to get the count of tr in training corpora in a language

        Pre-conditions: tr is a tri-gram character; language is a language from languages.

        Post-conditions: Returns count of tr in training corpora related to language (int).
        """

        corpora = self.trigrams_corpora[language]
        ct = corpora.get(tr, 0) #if not found count is 0
        return ct

    def predict_probs(self,path):
        """
        Method to show prediction probabilities of each language for each given sentence.

        Pre-conditions: Path is a string that represents either a path or a sentence

        Post-conditions: Returns vector of vector of probabilities associated to the sentences on the path file, or to the string associated. 
                        (array[float]).
        """
        
        try:
            preprocessed_test = self.__read_corpora(path)
        except:
            warnings.warn("Path not specified or incorrect, interpreting as string.")
            preprocessed_test = preprocessing(path)

        phrases = preprocessed_test.split("  ")
        predicted = []
        for phrase in phrases:
            phrase_probs = [(language, self.__likelihood(phrase, language)) for language in self.languages]
            predicted.append(phrase_probs)

        return predicted