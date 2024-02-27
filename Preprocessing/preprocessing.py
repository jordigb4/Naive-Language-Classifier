import re

"""
spa_path = 'corpora\spa_trn.txt'

with open(spa_path, 'r', encoding='utf-8') as file:
    
    spa_trn = file.read()
"""
def remove_digits(text):
    #Return: the string of text without digits

    return re.sub(r'\d+','',text)

def lower(text):
    #Return: the lowercase string from str

    return str.lower()

def reduce_whitespaces(text):
    #Return: string with only one whitespace between characters

    return re.sub(r'\s+', ' ', str)

def concatenate_whitespaces(text):
    #Return: string with all the phrases concatenated with a double space in the middle

    return re.sub(r'\n','  ', re.sub(r'\t','',text))

def preprocessing(text):
    return concatenate_whitespaces(reduce_whitespaces(lower(remove_digits(text))))


"""
raw_text = preprocessing(spa_trn)
print(raw_text[:5000])
"""