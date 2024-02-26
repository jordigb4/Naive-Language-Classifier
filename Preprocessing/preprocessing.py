import re

"""
spa_path = 'corpora\spa_trn.txt'

with open(spa_path, 'r', encoding='utf-8') as file:
    
    spa_trn = file.read()
"""
def remove_digits(text):
    #Return: the string of text without digits

    res = re.sub(r'\d+','',text)
    return res

def lower(str):
    #Return: the lowercase string from str

    return str.lower()

def reduce_whitespaces(str):
    #Return: string with only one whitespace between characters

    res = re.sub(r'\s+', ' ', str)
    return res

def concatenate_whitespaces(text):
    #Return: string with all the phrases concatenated with a double space in the middle

    res = re.sub(r'\n','  ', re.sub(r'\t','',text))
    return res

def preprocessing(text):
    return concatenate_whitespaces(reduce_whitespaces(lower(remove_digits(text))))


"""
raw_text = preprocessing(spa_trn)
print(raw_text[:5000])
"""