import re

def remove_digits(text: str) -> str:
    #Return: the string of text without digits

    return re.sub(r'\d+','',text)

def lower(text: str) -> str: 
    #Return: the lowercase string from str

    return text.lower()

def reduce_whitespaces(text: str) -> str:
    #Return: string with only one whitespace between characters
    return re.sub(r'[ ]+', ' ', text)

def concatenate_whitespaces(text: str) -> str:
    #Return: string with all the phrases concatenated with a double space in the middle

    return re.sub(r'\n','  ', re.sub(r'\t','',text))

def preprocessing(text:str) -> str:
    return concatenate_whitespaces(reduce_whitespaces(lower(remove_digits(text)))) 
