from imports import *
from args import Args
from models.retriever import PassageRetriever
from models.reader import QuestionAnswering

class Pipeline():
    def __init__(self, args):
        self.args = args
        
        file_path = os.path.join(self.args.data_dir, self.args.file_path)
        self.df = pd.read_csv(file_path)
        
        self.retriever = PassageRetriever(self.args, self.df)
        self.reader = QuestionAnswering(self.args, self.df)
        self.to(self.args.device)
                
    def __call__(self, questions, theme):
        rankings, scores = self.retriever.predict(questions, theme)
        outs = self.reader.predict(questions, rankings)
        for out,score in zip(outs,scores):
            out['retrieval_score'] = score[0]
            if out['answer']=='':
                out['Para_id'] = -1
        return outs

    def warmup(self):
        row = self.df.sample().iloc[0]
        _ = self([row.Question], row.Theme)
        return
    
    def time_profile(self, questions, theme):
        fin_latencies = {'total_latency': 0}
        
        start = time.time()
        rankings, scores,latencies = self.retriever.time_profile(questions, theme)
        end = time.time()
        fin_latencies.update(latencies)
        
        latencies['retrieval_latency'] = (end-start)*1000/len(questions)
        
        start = time.time()
        outs, latencies = self.reader.time_profile(questions, rankings)
        end = time.time()
        fin_latencies.update(latencies)
        
        latencies['reader_latency'] = (end-start)*1000/len(questions)
        
        for out, score in zip(outs,scores):
            out['retrieval_score'] = score[0]
            if out['answer']=='':
                out['Para_id'] = -1
        
        for e in fin_latencies.keys():
            fin_latencies['total_latency'] += fin_latencies[e]
        fin_latencies['retriever_latency'] = fin_latencies['total_latency'] - fin_latencies['reader_latency']
            
        return outs, fin_latencies
    
    def to(self, device):
        self.args.device = device
        self.retriever.to(device)
        self.reader.to(device)
        self.warmup()
    
    def dump(self, device='cpu'):
        self.load_torch()
        joblib_path = os.path.join(self.args.checkpoints_dir, self.args.joblib_path)
        self.to(device)
        joblib.dump(self, joblib_path)
        
    def load_torch(self):
        self.retriever.load_torch()
        self.reader.load_torch()
        self.warmup()
        
    def load_onnx(self):
        self.to('cpu')
        self.retriever.load_onnx()
        self.reader.load_onnx()
        self.warmup()
        
    @staticmethod
    def load_from_checkpoint(args, device='cpu'):
        joblib_path = os.path.join(args.checkpoints_dir, args.joblib_path)
        pipe = joblib.load(joblib_path)
        pipe.to(device)
        return pipe

if __name__=="__main__":    
    args = TrainingArgs()
    pipeline = Pipeline(args)