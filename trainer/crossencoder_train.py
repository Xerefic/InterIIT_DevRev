from imports import *

class ReRankerTrainer():
    def __init__(self, args, df, pipe):
        self.args = args
        self.base_model = CrossEncoder(self.args.themetrainer.ranker_model_name)
        self.to(self.args.device)
        self.fit(df, pipe)
        
    def to(self, device):
        self.args.device = device
        self.base_model._target_device = device
                    
    def collect_data(self, df, pipe, n_questions):
        data = []
        for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
            qns = row.GeneratedQuestions
            if len(row.GeneratedQuestions) > n_questions:
                qns = random.sample(row.GeneratedQuestions,n_questions)
            retrieved,_ = pipe.retriever.predict(qns,row.Theme)
            for i,qn in enumerate(qns):
                data.append({'Para_id':row.Para_id,
                             'Paragraph':row.Paragraph,
                             'Question':qn,
                             'Theme':row.Theme,
                             'retrieved':retrieved[i]})
        data = pd.DataFrame(data)
        return data
    
    def create_examples(self, data, para_lookup, n_negatives):
        examples = []
        for entry in tqdm.tqdm(data, disable=True):
            examples.append(InputExample(texts=[entry['Question'] , entry['Paragraph']], label=1))
            retrieved = entry['retrieved'].copy()
            if entry['Para_id'] in retrieved:
                retrieved.remove(entry['Para_id'])
            negative_ids = random.sample(retrieved, n_negatives)
            for idx in negative_ids:
                examples.append(InputExample(texts=[entry['Question'] , para_lookup[idx]], label= 0))
        return examples
    
    def save_onnx(self, model, path, do_lower_case=True, input_names=["input_ids", "attention_mask", "token_type_ids"]):
        providers = [self.args.onnx_provider]
        model.save('temp')
        time.sleep(2)
        configuration = AutoConfig.from_pretrained('temp', from_tf=False, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained('temp', do_lower_case=do_lower_case, from_tf=False, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained('temp', from_tf=False, config=configuration, local_files_only=True)
        os.system(f'rm -r temp')
        st = ["dummy"]

        inputs = tokenizer(st, padding=True, truncation='longest_first', 
                           max_length=tokenizer.__dict__["model_max_length"], return_tensors="pt",)
        
        model.eval()
        with torch.no_grad():

            symbolic_names = {0: "batch_size", 1: "max_seq_len"}

            torch.onnx.export(model, args=tuple(inputs.values()),
                f=path, opset_version=13,  # ONX version needs to be >= 13 for sentence transformers.
                do_constant_folding=True, input_names=input_names, output_names=["end"],
                dynamic_axes={
                    "input_ids": symbolic_names,
                    "token_type_ids": symbolic_names,
                    "attention_mask": symbolic_names,
                },
            )
            
    def fit(self, df, pipe):
        data = self.collect_data(df, pipe, self.args.themetrainer.n_questions)
        para_lookup = dict(zip(df.Para_id, df.Paragraph))
        for theme, G in tqdm.tqdm(data.groupby('Theme'), total=data.Theme.nunique()):
            examples = self.create_examples(G.to_dict('records'), para_lookup)
            n = int(len(examples) * self.args.themetrainer.train_frac)
            train = examples[:n]
            test = examples[n:]
            evaluator = CEBinaryClassificationEvaluator.from_input_examples(test, name='Testing')
            print(f'Theme: {theme} | Training Data Points = {len(train)}', end=' | ')
            train_dataloader = DataLoader(train, shuffle=True, batch_size=self.args.themetrainer.batch_size)
            warmup_steps = int(len(train_dataloader) * self.args.themetrainer.max_epochs * 0.1) #10% of train data for warm-up
            model = copy.deepcopy(self.base_model)
            score = evaluator(model)
            print(f'Score Before = {score}', end='  ')
            model.fit(train_dataloader=train_dataloader,
                      epochs=self.args.themetrainer.max_epochs,
                      warmup_steps=warmup_steps,
                      output_path=None)
            score = evaluator(model)
            print(f'Score After = {score}', end='\n\n')
            theme = re.sub("\.", "", theme)
            path = os.path.join(self.args.saves_path, f'crossencoder_{theme}.onnx')
            self.save_onnx(model, path)