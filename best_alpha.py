from language_identifier import LanguageIdentifier
from sklearn.metrics import accuracy_score

LangId = LanguageIdentifier()

alphas = [0.3, 0.4, 0.5, 0.6, 0.7]
languages = ["deu", "eng", "fra", "ita", "nld", "spa"]
for alpha in alphas:
    for lang in languages:

def alpha_score(clf: LanguageIdentifier, language:str, paths: list):
    y_ pred = 