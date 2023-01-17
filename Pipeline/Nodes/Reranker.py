from sentence_transformers import CrossEncoder
import numpy as np

class Reranker():
    def __init__(self,model=None,model_name='cross-encoder/ms-marco-MiniLM-L-4-v2'):
        self.model  = model or CrossEncoder(model_name)
        self.para_id_to_paras = None
        self.setup_flag = False
        
    def setup(self,df):
        if self.setup_flag: 
            print('Already setup is done! (Ranker)')
            return
        cols = ['Para_id','Paragraph']
        df = df.copy()
        assert df.columns.isin(cols).sum()==2 , "Verify Para_id, Paragraph are in the columns"
        df = df.loc[:,cols].drop_duplicates().reset_index(drop=True).set_index('Para_id')
        self.para_id_to_paras = dict(zip(df.index,df.Paragraph))
        self.setup_flag = True
        
    def __call__(self,questions,preds,batch=32,num_workers=0,**kwargs):
        assert self.para_id_to_paras is not None, 'Call setup function before reranking!'
        pairs = [(qn,self.para_id_to_paras[p])  for qn,pred in zip(questions,preds) for p in pred]
        scores = self.model.predict(pairs,batch_size=batch,num_workers=num_workers)
        indices = [0]+[len(pred) for pred in preds]
        indices = np.cumsum(indices)
        out = []
        for pred,i in zip(preds,range(1,len(indices))):
            score = scores[indices[i-1]:indices[i]]
            out.append([pred[idx] for idx in score.argsort()[::-1]])
        return out
    
    def to(self,device):
        self.model._target_device=device