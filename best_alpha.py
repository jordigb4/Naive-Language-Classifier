from language_identifier import LanguageIdentifier
from Preprocessing.preprocessing import preprocessing
from sklearn.metrics import accuracy_score
import numpy as np

def read_corpora(path):
        """
        Method to read and preprocess the corpora of a given path.

        Pre-conditions: path is a valid path to a readable file.

        Post-conditions: Returns the preprocessed file as raw text (str). 
        """
        with open(path, 'r', encoding='utf-8') as file:

            return preprocessing(file.read())
        
languages = ["deu", "eng", "fra", "ita", "nld", "spa"]
for language in languages:
    corpora = read_corpora(f"corpora/{language}_trn.txt")
    phrases = corpora.split("  ")[:10000]
    
    with open(f"corpora/{language}_val.txt", 'w', encoding='utf-8') as f:
        for line in phrases:
            f.write(line)

def alpha_score(clf: LanguageIdentifier, language:str, alpha: float):

    y_pred = clf.identify_language(path = f"corpora/{language}_val.txt", smoothing = "Lidstone", alpha = alpha)
    y_true = np.full(y_pred.shape[0], language)

    return accuracy_score(y_true, y_pred)

LangId = LanguageIdentifier()

alphas = [0.7, 0.8, 0.9]; acc = 0; best_alpha = None
for alpha in alphas:
    acc_alpha = np.mean([alpha_score(LangId, language, alpha) for language in languages])
    if acc_alpha > acc: 
        best_alpha = alpha
print(best_alpha)
