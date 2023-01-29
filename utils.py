from imports import *

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

stemmer = SnowballStemmer("english")
def clean_text(text, stem=False):
    text = text.replace(pronounciations_regex, '')
    text = unidecode.unidecode(text) # stips non-unidecode characters
    text = text.lower()
    text = re.sub(r'[^\x00-\x7F]', ' ',text)
    text = re.sub(r'[\\\'\"()]', '',text)
    text = re.sub('[_\-\?]', ' ',text)
    text = re.sub(r'\s+', ' ',text)
    if stem:
        text = ''.join([stemmer.stem(y) for y in text])
    return text

