from imports import *

class QuestionAnswering():
    def __init__(self, args, df):
        self.args = args
        if not self.args.reader.onnx:
            self.qa_cuda_support = True
            device = 0 if self.args.device=='cuda' else -1
            self.model = transformers_pipeline("question-answering", model=self.args.reader.model_name,
                                               tokenizer=self.args.reader.model_name, device=device)
        else:
            self.qa_cuda_support = False
            self.model = self.load_optimized()
        self.fit(df)
        
        
    def load_optimized(self):
        model = ORTModelForQuestionAnswering.from_pretrained(self.args.reader.model_name, from_transformers=True) # Load onnx model
        tokenizer = AutoTokenizer.from_pretrained(self.args.reader.model_name)
        
        optimizer = ORTOptimizer.from_pretrained(model)
        optimization_config = OptimizationConfig(optimization_level=99)
        optimizer.optimize(save_dir='temp', optimization_config=optimization_config)
        
        model = ORTModelForQuestionAnswering.from_pretrained('temp')
        # os.system('rm -r temp')
        pipe = opt_pipeline("question-answering", model=model, tokenizer=tokenizer)
        return pipe
            
    def fit(self, df):
        paras = df.loc[:,['Para_id','Paragraph']].drop_duplicates().set_index('Para_id',drop=True)
        self.para_lookup = dict(zip(paras.index,paras.Paragraph))
        
    def __call__(self, qns, retrieved):
        context = [self.para_lookup[retrd[0]] for retrd in retrieved]
        out = self.model(question=qns, context=context, handle_impossible_answer=True, 
                         batch_size=self.args.reader.batch_size, max_seq_len=self.args.reader.max_seq_len)
        if isinstance(out, dict): out = [out]
        for entry, retrieves in zip(out,retrieved):
            entry['Para_id'] = retrieves[0]
        if not isinstance(out, list): out = [out]
        out_df = pd.DataFrame(out)
        return out_df
    
    # def to(self):
    #     if self.args.device=='cuda': device=0
    #     if self.qa_cuda_support:
    #         model = self.model.model.to('cpu')
    #         tokenizer = self.model.tokenizer
    #         self.model = pipeline("question-answering", model=model, tokenizer=tokenizer)