from imports import *
from utils import clean_text

class DenseRetriever():
    def __init__(self, args, df):
        self.args = args
        self.model_Q = SentenceTransformer(self.args.retriever.dpr_model_Q).to(self.args.device)
        self.model_P = SentenceTransformer(self.args.retriever.dpr_model_P).to(self.args.device)
        self.model_Q.eval()
        self.model_P.eval()
        
        self.fit(df)
    
    def fit(self, df):
        df = copy.deepcopy(df)
        cols = ['Para_id', 'Paragraph', 'Theme']
        assert df.columns.isin(cols).sum()==3 , "Verify Para_id, Paragraph, Theme are in the columns"
        
        paras = df.loc[:,cols].drop_duplicates().reset_index(drop=True)
        self.idx_to_para_id = dict(zip(paras.index, paras.Para_id))
        paras.loc[:,'Paragraph'] = paras.Paragraph.apply(lambda x: clean_text(x, stem=False))
        self.embeddings = self.model_P.encode(paras.Paragraph,show_progress_bar=True)
        self.theme_to_idxs = paras.index.to_series().groupby(paras.Theme).unique()
        
    def predict(self, questions, theme):
        questions = [clean_text(q, stem=False) for q in questions]
        dense_qns = self.model_Q.encode(questions, show_progress_bar=False)
        idx = self.theme_to_idxs[theme].tolist()
        top_k = self.args.retriever.dpr_top_k
        
        embeds = self.embeddings[idx]
        rankings = semantic_search(dense_qns,embeds,top_k=top_k)
        scores = [ [ rank['score']  for rank in ranks] for ranks in rankings ]
        rankings = [ [ self.idx_to_para_id[idx[rank['corpus_id']]]  for rank in ranks] for ranks in rankings ]
        return rankings, scores