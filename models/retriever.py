from .retrievers import *
from imports import *

class PassageRetriever():
    def __init__(self, args, df):
        self.args = args
        self.weights = dict(self.args.retriever.weights)
        threshold_path = os.path.join(self.args.saves_path, self.args.retriever.ranker_treshold_path)
        if os.path.exists(threshold_path):
            with open(threshold_path) as f:
                self.threshold = json.load(f)
        else:
            self.threshold = {}
        
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
        if self.args.retriever.reranker=='cross-encoder':
            self.ranker = ReRanker(self.args, df)
        
    def predict(self, questions, theme):
        rankings, scores, weights = [], [], []
        for e in self.retrievers.keys():
            ranking, score = self.retrievers[e].predict(questions, theme)
            rankings.append(ranking)
            scores.append(score)
            weights.append(self.weights[e])
        rankings, scores = self.voter.predict(rankings, scores, weights)
        if self.args.retriever.use_ranker:
            to_pred_idx = []
            for i, score in enumerate(scores):
                diff = score[0]-score[1]
                if diff < self.threshold.get(theme, 0.1): 
                    to_pred_idx.append(i)
            if len(to_pred_idx)>0:
                _rankings = [rankings[i] for i in to_pred_idx]
                _questions = [questions[i] for i in to_pred_idx]
                _rankings,_scores = self.ranker.predict(_questions, _rankings)
            for i, idx in enumerate(to_pred_idx):
                rankings[idx] = _rankings[i]
                scores[idx] = _scores[i]
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
        start = time.time()
        fin_rankings, fin_scores = self.ranker.predict(questions, rankings)
        end = time.time()
        latencies['ranker_latency'] = (end-start)*1000/len(questions)
        
        return fin_rankings, fin_scores, latencies
    
    def to(self, device):
        for e in self.retrievers.keys():
            self.retrievers[e].to(device)
        self.ranker.to(device)
            
    def load_torch(self):
        for e in self.retrievers.keys():
            self.retrievers[e].load_torch()
        self.ranker.load_torch()
            
    def load_onnx(self):
        for e in self.retrievers.keys():
            self.retrievers[e].load_onnx()
        self.ranker.load_onnx()