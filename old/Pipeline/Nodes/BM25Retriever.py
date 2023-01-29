from .Helpers.ES import ElasticSearchAPI
from .Helpers.dfCleaner import clean_text
class BM25Retriever:
    def __init__(self,es=None,es_kwargs={}):
        self.es = es if es is not None else ElasticSearchAPI(**es_kwargs)
        self.setup_flag = False
        
    def setup(self,df):
        if self.setup_flag: 
            print('Already setup is done! (ES)')
            return
        df = df.copy() # Prevent inplace modifications!
        cols = ['Para_id','Paragraph','Theme']
        assert df.columns.isin(cols).sum()==3 , "Verify Para_id, Paragraph, Theme are in the columns"
        paras = df.loc[:,cols].drop_duplicates().reset_index(drop=True)
        paras.loc[:,'Paragraph'] = paras.Paragraph.apply(clean_text)
        data = paras.to_dict('index') # Para_id == _id
        self.es.batch_upload(data)
        self.setup_flag = True
    
    def __call__(self,questions,theme,top_k=30,_source=True):
        assert self.setup_flag, 'Setup is not ran yet!'
        res = self.es.batch_search(questions,themes=[theme] * len(questions)
                                     ,top_k=top_k,_source=_source)
        res = [[elem for elem in preds if elem['Theme']==theme] for preds in res] # Remove partial match on themes
        scores = [[e['_score'] for e in elems]  for elems in res]
        res = [[e['Para_id'] for e in elems]  for elems in res]
        return res, scores      