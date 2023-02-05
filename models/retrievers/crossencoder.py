from imports import *

class ReRanker():
    def __init__(self, args, df):
        self.args = args
        self.df = df
        self.backend = None
        if self.args.device=='cpu':
            self.load_torch()
            # self.load_onnx()
        elif self.args.device=='cuda':
            self.load_torch()
        
        self.fit(df)
        
    def fit(self, df):
        cols = ['Para_id','Paragraph']
        df = copy.deepcopy(df)
        assert df.columns.isin(cols).sum()==2 , "Verify Para_id, Paragraph are in the columns"
        
        df = df.loc[:,cols].drop_duplicates().reset_index(drop=True).set_index('Para_id')
        self.para_id_to_paras = dict(zip(df.index, df.Paragraph))
        
    def predict(self, questions, rankings, **kwargs):
        # if theme:
        #     model = self.get_theme_model(theme)
        # else:
        #     model = self.model
        batch_size = self.args.retriever.ranker_batch_size
        pairs = [(q, self.para_id_to_paras[p])  for q, ranking in zip(questions, rankings) for p in ranking]
        scores = self.model.predict(pairs, batch_size=batch_size)
        indices = [0]+[len(pred) for pred in rankings]
        indices = np.cumsum(indices)
        fin_rankings, fin_scores = [], []
        for pred, i in zip(rankings, range(1, len(indices))):
            score = scores[indices[i-1]:indices[i]]
            sort_idx = score.argsort()[::-1]
            fin_scores.append([score[idx] for idx in sort_idx])
            fin_rankings.append([pred[idx] for idx in sort_idx])
        return fin_rankings, fin_scores
    
    def to(self, device):
        if self.backend=='torch':
            self.args.device = device
            self.model._target_device = device
            
    def load_theme_model(self,theme,backend='onnx'):
        backend = 'pth' if backend=='torch' else backend
        model_path = os.path.join(self.args.saves_path, f'crossencoder_{theme}.{backend}')
        if not os.path.exists(model_path):return
        if backend=='onnx':
            self.model = OnnxRanker(session=onnxruntime.InferenceSession(model_path, onnxruntime.SessionOptions(),
                                                              providers=[self.args.onnx_provider]), tokenizer = self.model.tokenizer, args=self.args)
            print(f'loaded {theme}')
        
        if backend=='torch':
            self.model = CrossEncoder(model_path)
            print(f'loaded {theme}')
            
    def get_theme_model(self, theme):
        theme = re.sub("\.", "", theme)
        if self.backend=='torch':
            model_path = os.path.join(self.args.saves_path, f'crossencoder_{theme}.pth')
            if os.path.exists(model_path):
                theme_model = CrossEncoder(model_path)
            else:
                theme_model = self.model
            self.to(self.args.device)
        elif self.backend=='onnx':
            tokenizer = self.model.tokenizer
            model_path = os.path.join(self.args.saves_path, f'crossencoder_{theme}.onnx')
            if os.path.exists(model_path):
                theme_model = OnnxRanker(session=onnxruntime.InferenceSession(model_path, onnxruntime.SessionOptions(),
                                                              providers=[self.args.onnx_provider]), tokenizer=tokenizer, args=self.args)
                print('loaded theme')
            else:
                theme_model = self.model
        return theme_model
            
    def load_torch(self):
        if self.backend!='torch':
            self.backend = 'torch'
            self.model = CrossEncoder(self.args.retriever.ranker_model_name)
            self.to(self.args.device)
            self.model.model.eval()
            
    def load_onnx(self):
        if self.backend!='onnx':
            self.backend = 'onnx'
            self.model = sentence_transformers_onnx_ranker(self.args, self.model, 'temp')
            os.system('rm -rf temp')
            os.system('rm temp.onnx')
    

class OnnxRanker:
    """
    Credits: https://github.com/UKPLab/sentence-transformers/issues/46
    OnnxEncoder dedicated to run SentenceTransformer under OnnxRuntime.
    """

    def __init__(self, session, tokenizer, args):
        self.args = args
        self.session = session
        self.tokenizer = tokenizer
        self.max_length = tokenizer.__dict__["model_max_length"]
        
    def smart_batching_collate_text_only(self, batch):
        texts = [[] for _ in range(len(batch[0]))]

        for example in batch:
            for idx, text in enumerate(example):
                texts[idx].append(text.strip())

        tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)
        return tokenized

    def predict(self, sentences, **kwargs):
        batch_size =  self.args.retriever.ranker_batch_size
        
        iterator = DataLoader(sentences, batch_size=batch_size, 
                                     collate_fn=self.smart_batching_collate_text_only, shuffle=False)
        activation = nn.Sigmoid()
        global pred_scores
        pred_scores = []
        for features in iterator:
            features = {key:value.numpy() for key,value in features.items()}
            model_predictions = self.session.run(None, features)[0]
            model_predictions = torch.tensor(model_predictions)
            logits = activation(model_predictions)
            pred_scores.extend(logits)
        return torch.concat(pred_scores).numpy()
                
    
def sentence_transformers_onnx_ranker(args, model, path, do_lower_case=True, input_names=["input_ids", "attention_mask", "token_type_ids"]):
        providers = [args.onnx_provider]
        model.save(path)
        configuration = AutoConfig.from_pretrained(path, from_tf=False, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(path, do_lower_case=do_lower_case, from_tf=False, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(path, from_tf=False, config=configuration, local_files_only=True)

        st = ["dummy"]

        inputs = tokenizer(st, padding=True, truncation='longest_first', 
                           max_length=tokenizer.__dict__["model_max_length"], return_tensors="pt",)
        
        model.eval()
        with torch.no_grad():

            symbolic_names = {0: "batch_size", 1: "max_seq_len"}

            torch.onnx.export(model, args=tuple(inputs.values()),
                f=f"{path}.onnx", opset_version=13,  # ONX version needs to be >= 13 for sentence transformers.
                do_constant_folding=True, input_names=input_names, output_names=["end"],
                dynamic_axes={
                    "input_ids": symbolic_names,
                    "token_type_ids": symbolic_names,
                    "attention_mask": symbolic_names,
                },
                operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
            )
            _model = onnx.load(f"{path}.onnx")
            # model_simp, check = onnx_simplify(_model)
            if args.onnx_float16:
                _model = convert_float_to_float16(_model)
            onnx.save(_model,f"{path}.onnx")

            sess_options = onnxruntime.SessionOptions()
            # sess_options.execution_mode  = onnxruntime.ExecutionMode.ORT_PARALLEL
            # sess_options.inter_op_num_threads = args.n_jobs
            return OnnxRanker(session=onnxruntime.InferenceSession(f"{path}.onnx", sess_options, providers=providers),
                tokenizer=tokenizer, args=args)