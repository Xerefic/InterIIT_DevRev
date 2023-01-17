from sentence_transformers.util import semantic_search
from sentence_transformers import SentenceTransformer
from .Helpers.dfCleaner import *

class DenseRetriever():
    def __init__(self,model_Q=None,
                 model_P=None,
                 model_Q_name='flax-sentence-embeddings/multi-QA_v1-mpnet-asymmetric-Q',
                 model_P_name='flax-sentence-embeddings/multi-QA_v1-mpnet-asymmetric-A',
                 device='cpu'):
        self.model_Q = model_Q or SentenceTransformer(model_Q_name)
        self.model_P = model_P or SentenceTransformer(model_P_name)
        self.to(device)
        self.idx_to_para_id = None
        self.theme_to_idxs = None
        self.embeddings = None
        self.setup_flag= False
    
    def setup(self,df):
        if self.setup_flag: 
            print('Already setup is done! (Dense)')
            return
        df = df.copy() # Prevent inplace modifications!
        cols = ['Para_id','Paragraph','Theme']
        assert df.columns.isin(cols).sum()==3 , "Verify Para_id, Paragraph, Theme are in the columns"
        
        paras = df.loc[:,cols].drop_duplicates().reset_index(drop=True)
        
        self.idx_to_para_id = dict(zip(paras.index,paras.Para_id))
             
        paras.loc[:,'Paragraph'] = paras.Paragraph.apply(clean_dense)
        self.embeddings = self.model_P.encode(paras.Paragraph,show_progress_bar=True)
        self.theme_to_idxs = paras.index.to_series().groupby(paras.Theme).unique()
        self.model_P = None
        self.setup_flag = True
        
    def __call__(self,questions,theme,top_k=30,pbar=False):
        dense_qns = self.model_Q.encode(questions,show_progress_bar= pbar)
        idx = self.theme_to_idxs[theme].tolist()
        embeds = self.embeddings[idx]
        rankings = semantic_search(dense_qns,embeds,top_k=top_k)
        scores = [ [ rank['score']  for rank in ranks] for ranks in rankings ]
        rankings = [ [ self.idx_to_para_id[idx[rank['corpus_id']]]  for rank in ranks] for ranks in rankings ]
        return rankings,scores
    
    def to(self,device):
        if self.model_P: self.model_P._target_device = device
        if self.model_Q: self.model_Q._target_device = device