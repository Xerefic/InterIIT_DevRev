import nltk;nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import unidecode
import re

stemmer = SnowballStemmer("english")
stop = stopwords.words('english')
pat = r'\b(?:{})\b'.format('|'.join(stop))



def clean_df(df,cols):
    for col in cols:
        df.loc[:,col] = df.loc[:,col].str.replace(pronounciations_regex, '',regex=True)
        df.loc[:,col] = df.loc[:,col].apply(unidecode.unidecode)
        df.loc[:,col] = df.loc[:,col].str.lower()
        df.loc[:,col] = df.loc[:,col].apply(strip_stop)
        df.loc[:,col] = df.loc[:,col].str.replace(r'[^\x00-\x7F]', ' ',regex=True)
        df.loc[:,col] = df.loc[:,col].str.replace(r'[\\\'\"()]', '',regex=True)
        df.loc[:,col] = df.loc[:,col].str.replace('[_\-\?]', ' ',regex=True)
        df.loc[:,col] = df.loc[:,col].str.replace(r'\s+', ' ',regex=True)
        df.loc[:,col] = df.loc[:,col].apply(lambda x: ''.join([stemmer.stem(y) for y in x]))
    return df

def clean_text(text):
    text = text.replace(pronounciations_regex, '')
    text = unidecode.unidecode(text)
    text = text.lower()
    text = strip_stop(text)
    text = re.sub(r'[^\x00-\x7F]', ' ',text)
    text = re.sub(r'[\\\'\"()]', '',text)
    text = re.sub('[_\-\?]', ' ',text)
    text = re.sub(r'\s+', ' ',text)
    text = ''.join([stemmer.stem(y) for y in text])
    return text

def clean_dense(text):
    text = text.replace(pronounciations_regex, '')
    text = unidecode.unidecode(text)
    text = text.lower()
    text = re.sub(r'[^\x00-\x7F]', ' ',text)
    text = re.sub(r'[\\\'\"()]', '',text)
    text = re.sub('[_\-\?]', ' ',text)
    text = re.sub(r'\s+', ' ',text)
    return text

def strip_stop(text):
    return re.sub(pat,'',text)

pronounciations_regex = r'\s*\((((.*[\u0250-\u02AF]+)|(.*[\u02B0–\u02FF]+)|(.*[\u1D00–\u1D7F]+)|(.*[\u1D80–\u1DBF]+)|(.*[\uA700–\uA71F]+)|(.*[\u2070–\u209F]+))\/).*?\)'

# cols = ['Question','Paragraph','Theme']