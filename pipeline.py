from imports import *
from args import TrainingArgs
from models.retriever import PassageRetriever
from models.reader import QuestionAnswering

class Pipeline():
    def __init__(self, args):
        self.args = args
        
        file_path = os.path.join(self.args.data_dir, self.args.file_path)
        self.df = pd.read_csv(file_path)
        
        self.retriever = PassageRetriever(self.args, self.df)
        self.reader = QuestionAnswering(self.args, self.df)
        
    def __call__(self, questions, theme):
        rankings, scores = self.retriever.predict(questions, theme)
        outs = self.reader.predict(questions, rankings)
        for out in outs:
            if out['answer']=='':
                out['Para_id'] = -1
        return outs
    
    def to(self, device):
        self.retriever.to(device)
        self.reader.to(device)
    
    def dump(self, device='cpu'):
        joblib_path = os.path.join(self.args.checkpoints_dir, self.args.joblib_path)
        self.to(device)
        joblib.dump(self, joblib_path)

if __name__=="__main__":    
    
    args = TrainingArgs()
    pipeline = Pipeline(args)