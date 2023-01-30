from imports import *

class Voter:
    def __init__(self, args, df):
        self.args = args
        self.fit(df)
    
    def fit(self, df):
        pass 
    
    def predict(self, rankings, scores, weights):
        fin_scores = []
        fin_rankings = []
        top_k = self.args.retriever.voting_top_k      
        for idx in range(len(rankings[0])):
            scores_map = defaultdict(int)
            for out, score, weight in zip(rankings, scores, weights):
                if score[idx]:
                    scale = max(score[idx]) - min(score[idx]) + 1e-12
                    shift = min(score[idx]) 
                for o, s in zip(out[idx], score[idx]):
                    scores_map[o] += (s-shift)/scale * weight
            entry = list(scores_map.keys())
            entry.sort(key=lambda x:scores_map[x], reverse=True)
            fin_rankings.append(entry[:top_k])
            fin_scores.append([scores_map[_id] for _id in entry[:top_k]])
        return fin_rankings, fin_scores