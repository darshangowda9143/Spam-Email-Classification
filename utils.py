import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text, debug=False):
    steps = {}
    
    # 0. Original
    if debug: steps['original'] = text

    # 1. Lowercase
    text = text.lower()
    if debug: steps['lowercase'] = text
    
    # 2. Tokenize
    text = nltk.word_tokenize(text)
    if debug: steps['tokenized'] = list(text)

    # 3. Alphanumeric
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    if debug: steps['alphanumeric'] = list(text)
    y.clear()

    # 4. Stopwords & Punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    if debug: steps['no_stopwords'] = list(text)
    y.clear()

    # 5. Stemming
    for i in text:
        y.append(ps.stem(i))
    
    if debug: steps['stemmed'] = list(y)

    final_text = " ".join(y)
    
    if debug:
        return final_text, steps
    return final_text
