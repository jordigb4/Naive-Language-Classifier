from string import digits
import re

spa_path = 'corpora\spa_trn.txt'

with open(spa_path, 'r', encoding='utf-8') as file:
    
    spa_trn = file.read()

def remove_digits(text):

    res = re.sub(r'\d+','',text)
    return res

def concatenate_whitespaces(text):
    res = re.sub(r'\n','  ',re.sub(r'\t','',text))

    return res 
print(repr(concatenate_whitespaces(remove_digits(spa_trn[:5000]))))