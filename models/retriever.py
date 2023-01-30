from .retrievers import *

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
    
    def to(self, device):
        for e in self.retrievers.keys():
            self.retrievers[e].to(device)