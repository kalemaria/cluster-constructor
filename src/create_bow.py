import unicodedata
import re
from sklearn.feature_extraction.text import CountVectorizer

'''This set of functions helps to create Bag Of Words (BOW) from a text corpus.'''

def remove_accented_chars(text):
    '''
    Removes accented characters from a text.

    Parameters:
    - text (str): Input text.

    Returns:
    - str: Text with accented characters removed.
    '''
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def pre_process_corpus(docs):
    '''
    Preprocesses a list of documents in a text corpus.

    Parameters:
    - docs (list): List of documents (strings).

    Returns:
    - list: Normalized and preprocessed documents.
    '''
    norm_docs = []
    for doc in docs:
        doc = doc.translate(doc.maketrans("\n\t\r", "   "))
        doc = doc.lower()
        doc = remove_accented_chars(doc)
        # lower case and remove special characters\whitespaces
        doc = re.sub(r'[^a-zA-Z0-9\s]', ' ', doc, flags=re.I|re.A)
        doc = re.sub(' +', ' ', doc)
        doc = doc.strip()
        norm_docs.append(doc)
    return norm_docs

def create_bow(docs):
    '''
    Creates a Bag of Words (BOW) representation from a list of documents.

    Parameters:
    - docs (list): List of documents (strings).

    Returns:
    - scipy.sparse.csr_matrix: BOW representation of the documents.
    '''
    docs = pre_process_corpus(docs)
    cv = CountVectorizer()
    cv_features = cv.fit_transform(docs)
    return cv_features