from .Nodes.BM25Retriever import BM25Retriever
from .Nodes.DenseRetriever import DenseRetriever
from .Nodes.Reranker import Reranker
from .Nodes.QA import QA
from collections import defaultdict
import joblib
from time import time
class Pipe():
    def __init__(self,saved_path=None,bm25=None,dense=None,ranker=None,qa=None,
                bm25_kwargs={},dense_kwargs={},
                ranker_kwargs={},qa_kwargs={},load_kwargs={}):
        
        self.bm25 = bm25 or BM25Retriever(**bm25_kwargs)
        if saved_path:
            objs = joblib.load(saved_path,**load_kwargs)
            for key,value in objs.items():
                self.__setattr__(key,value)
                print(f'Loaded {key}')
            if 'dense' not in objs: self.dense = dense or DenseRetriever(**dense_kwargs)
            if 'ranker' not in objs: self.ranker = ranker or Reranker(**ranker_kwargs)
            if 'qa' not in objs: self.qa = qa or QA(**qa_kwargs)
            self.sample = None
            return
        self.dense = dense or DenseRetriever(**dense_kwargs)
        self.ranker = ranker or Reranker(**ranker_kwargs)
        self.qa = qa or QA(**qa_kwargs)
        self.sample = None
        
        
    def setup(self,df):
        self.sample = ([f'What is the topic of the article {df.iloc[0].Theme} ']*2,df.iloc[0].Theme)
        self.bm25.setup(df)
        self.dense.setup(df)
        self.ranker.setup(df)
        self.qa.setup(df)
        self(*self.sample)
        
    def __call__(self,qns,theme,bm25_kwargs={},dense_kwargs={},
                ranker_kwargs={},qa_kwargs={},voting_kwargs={},bm25=True,
            dense=True,ranker=True,qa=True):
    
        out1,scores1 = self.bm25(qns,theme,**bm25_kwargs) if bm25 else []
        out2,scores2 = self.dense(qns,theme,**dense_kwargs) if dense else []
        out = self.voting(out1,out2,scores1,scores2,**voting_kwargs)
        out = self.ranker(qns,out,**ranker_kwargs) if ranker else out
        out = self.qa(qns,out,**qa_kwargs) if qa else out
        return out
    
    def test(self,qns,theme,bm25_kwargs={},dense_kwargs={},
                ranker_kwargs={},qa_kwargs={},voting_kwargs={},bm25=True,
            dense=True,ranker=True,qa=True):
        
        stats = {}
        t = time()
        out1,scores1 = self.bm25(qns,theme,**bm25_kwargs) if bm25 else []
        t = time()-t
        stats['latency_bm25(ms)/qn'] = t/len(qns) * 1000
        
        t = time()
        out2,scores2 = self.dense(qns,theme,**dense_kwargs) if dense else []
        t = time()-t
        stats['latency_dense(ms)/qn'] = t/len(qns) * 1000
        
        t = time()
        out = self.voting(out1,out2,scores1,scores2,**voting_kwargs)
        t = time()-t
        stats['Voting(ms)/qn'] = t/len(qns) * 1000
        
        t = time()
        out = self.ranker(qns,out,**ranker_kwargs) if ranker else out
        t = time()-t
        stats['latency_ranker(ms)/qn'] = t/len(qns) * 1000
        
        t = time()
        out = self.qa(qns,out,**qa_kwargs)
        t = time()-t
        stats['latency_qa(ms)/qn'] = t/len(qns) * 1000 if qa else out
        return out,stats
        
    
    def to(self,device):
        self.dense.to(device)
        self.ranker.to(device)
        self.qa.to(device)
            
        if self.sample:
            self(*self.sample)
            print(f'Moved to {device}!')
        else:
            print(f'Pipe will be moved to {device} completely on the next forward call or setup call.')
    
    def voting(self,out1,out2,score1,score2,top_k=5,w1=0.4,w2=0.6):
        out = []
        for idx in range(len(out1)):
            score = defaultdict(int)
            if score1[idx]:
                scale1 = max(score1[idx]) - min(score1[idx]) + 1e-12
                shift1 = min(score1[idx]) 
            if score2[idx]:
                scale2 = max(score2[idx]) - min(score2[idx]) + 1e-12
                shift2 = min(score2[idx]) 
            for o1,s1 in zip(out1[idx],score1[idx]):
                score[o1] += (s1-shift1)/scale1 * w1
            for o2,s2 in zip(out2[idx],score2[idx]):
                score[o2] += (s2-shift2)/scale2 * w2
            entry = list(score.keys())
            # print(score)
            entry.sort(key=lambda x:score[x],reverse=True)
            out.append(entry[:top_k])
        return out
    
    def save(self,path,to_save=['dense','ranker','qa'],**kwargs):
        to_save = {obj:self.__getattribute__(obj) for obj in to_save}
        # obj = (self.dense,self.ranker,self.qa)
        joblib.dump(to_save,path,**kwargs)
        