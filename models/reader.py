from imports import *

class QuestionAnswering():
    def __init__(self, args, df):
        self.args = args        
        if self.args.device=='cpu':
            self.load_onnx()
        elif self.args.device=='cuda':
            self.load_torch()
        
        self.fit(df)
            
    def fit(self, df):
        paras = df.loc[:,['Para_id','Paragraph']].drop_duplicates().set_index('Para_id',drop=True)
        self.para_lookup = dict(zip(paras.index,paras.Paragraph))
         
    def predict(self, questions, rankings):
        context = [self.para_lookup[retrieved[0]] for retrieved in rankings]
        out = self.model(question=questions, context=context, handle_impossible_answer=True, 
                         batch_size=self.args.reader.batch_size, max_seq_len=self.args.reader.max_seq_len)
        if isinstance(out, dict): out = [out]
        for entry, retrieves in zip(out,rankings):
            entry['Para_id'] = retrieves[0]
        if not isinstance(out, list): out = [out]
        return out
        # out_df = pd.DataFrame(out)
        # return out_df

    def time_profile(self, questions, rankings):
        latencies = {}
        
        start = time.time()
        outs = self.predict(questions,rankings)
        end = time.time()
        
        latencies['reader_latency'] = (end-start)*1000/len(questions)
        
        return outs, latencies
    
    def to(self, device):
        if self.backend=='torch':
            device = 0 if device=='cuda' else -1
            model_name = self.args.reader.model_name
            self.model = transformers_pipeline("question-answering", model=model_name, tokenizer=model_name, device=device)
            self.model(question=['dummy'], context=['dummy'], handle_impossible_answer=True)
            
    def load_torch(self):
        if self.backend=='onnx':
            self.backend = 'torch'
            model_name = self.args.reader.model_name
            device = 0 if device=='cuda' else -1
            self.model = transformers_pipeline("question-answering", model=model_name, tokenizer=model_name, device=device)
        
    def load_onnx(self):
        if self.backend=='torch':
            self.backend = 'onnx'
            model = ORTModelForQuestionAnswering.from_pretrained(self.args.reader.model_name, from_transformers=True) # Load onnx model
            tokenizer = AutoTokenizer.from_pretrained(self.args.reader.model_name)

            optimizer = ORTOptimizer.from_pretrained(model)
            optimization_config = OptimizationConfig(optimization_level=99)
            optimizer.optimize(save_dir='temp', optimization_config=optimization_config)

            model = ORTModelForQuestionAnswering.from_pretrained('temp')
            os.system('rm -r temp')
            pipe = optimum_pipeline("question-answering", model=model, tokenizer=tokenizer)
            self.model = pipe