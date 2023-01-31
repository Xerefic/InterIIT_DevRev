from imports import *
from utils import clean_text

class SparseRetriever():
    def __init__(self, args, df):
        self.args = args
        self.fit(df)
        

    def fit(self, df):
        df = copy.deepcopy(df)
        if 'GeneratedQuestions' in df.columns:
            df.Paragraph = df.Paragraph + ' ' + df.GeneratedQuestions.apply(lambda x: ' '.join(x))
        
        cols = ['Para_id', 'Paragraph', 'Theme']
        assert df.columns.isin(cols).sum()==3 , "Verify Para_id, Paragraph, Theme are in the columns"
    
        paras = df.loc[:, cols].drop_duplicates().reset_index(drop=True)
        paras.loc[:, 'Paragraph'] = paras.Paragraph.apply(lambda x: clean_text(x, stem=True))
        self.index = {}
        for theme, G in paras.groupby('Theme'):
            G['id'] = G.Para_id
            G['text'] = G.Paragraph.apply(clean_text)
            self.index[theme] = SearchEngine(theme, stopwords=None, stemmer=None, 
                                        hyperparams={'k1': self.args.retriever.k1, 'b': self.args.retriever.b})
            self.index[theme].index(G.drop(['Theme', 'Para_id', 'Paragraph'], 
                                           axis=1).to_dict('records'),show_progress=False)
            

    def predict(self, questions, theme):
        qns = [{'id': i,'text': qn} for i, qn in enumerate(questions)]
        out = self.index[theme].msearch(qns, cutoff=self.args.retriever.sparse_top_k)
        out = list(out.values())
        scores = [list(o.values()) for o in out]
        rankings = [list(o.keys()) for o in out]
        return rankings, scores
    
    def to(self, device):
        pass
    
