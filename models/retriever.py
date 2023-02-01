from .retrievers import *
from imports import *

class PassageRetriever():
    def __init__(self, args, df):
        self.args = args
        self.weights = dict(self.args.retriever.weights)
        
        self.retrievers = {}
        for e in self.args.retriever.ensemble:
            if e=='bm25':
                self.retrievers[e] = SparseRetriever(self.args, df)
            elif e=='dpr':
                self.retrievers[e] = DenseRetriever(self.args, df)
            elif e=='colbert':
                self.retrievers[e] = ColBertRetriever(self.args, df)
            else:
                raise NotImplementedError(f'{e} not implemented!')
            
        self.voter = Voter(self.args, df)
        
    def predict(self, questions, theme):
        rankings, scores, weights = [], [], []
        for e in self.retrievers.keys():
            ranking, score = self.retrievers[e].predict(questions, theme)
            rankings.append(ranking)
            scores.append(score)
            weights.append(self.weights[e])
        rankings, scores = self.voter.predict(rankings, scores, weights)
        return rankings, scores
    
    def time_profile(self,questions,theme):
        rankings, scores, weights, latencies = [], [], [], {}
        for e in self.retrievers.keys():
            
            start = time.time()
            ranking, score = self.retrievers[e].predict(questions, theme)
            end = time.time()
            
            rankings.append(ranking)
            scores.append(score)
            latencies[f'{e}_latency'] = (end-start)*1000/len(questions)
            weights.append(self.weights[e])
        start = time.time()
        rankings, scores = self.voter.predict(rankings, scores, weights)
        end = time.time()
        latencies['voting_latency'] = (end-start)*1000/len(questions)
        return rankings, scores, latencies
    
    def to(self, device):
        for e in self.retrievers.keys():
            self.retrievers[e].to(device)
            
    def load_torch(self):
        for e in self.retrievers.keys():
            self.retrievers[e].load_torch()
            
    def load_onnx(self):
        for e in self.retrievers.keys():
            self.retrievers[e].load_onnx()