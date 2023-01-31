from imports import *

class QuestionAnswering():
    def __init__(self, args, df):
        self.args = args
        
        self.load_model(model_name=self.args.reader.model_name)
        self.fit(df)
        
    def load_model(self, model_name):
        if not self.args.reader.onnx:
            self.cuda_support = True
            device = 0 if self.args.device=='cuda' else -1
            self.model = transformers_pipeline("question-answering", model=model_name, tokenizer=model_name, device=device)
        else:
            self.cuda_support = False
            self.model = self.load_optimized()    
        
    def load_optimized(self):
        model = ORTModelForQuestionAnswering.from_pretrained(self.args.reader.model_name, from_transformers=True) # Load onnx model
        tokenizer = AutoTokenizer.from_pretrained(self.args.reader.model_name)
        
        optimizer = ORTOptimizer.from_pretrained(model)
        optimization_config = OptimizationConfig(optimization_level=99)
        optimizer.optimize(save_dir='temp', optimization_config=optimization_config)
        
        model = ORTModelForQuestionAnswering.from_pretrained('temp')
        os.system('rm -r temp')
        pipe = opt_pipeline("question-answering", model=model, tokenizer=tokenizer)
        return pipe
            
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

    def time_profile(self,questions,rankings):
        latencies = {}
        
        start = time.time()
        outs = self.predict(questions,rankings)
        end = time.time()
        
        latencies['reader_latencies'] = (end-start)*1000/len(questions)
        
        return outs, latencies
    
    def to(self, device):
        if self.cuda_support and device!='cpu':
            device = -1
            model = self.model.model
            tokenizer = self.model.tokenizer
            self.model = transformers_pipeline("question-answering", model=model, tokenizer=tokenizer, device=device)
        self.model(question=['dummy'], context=['dummy'], handle_impossible_answer=True)