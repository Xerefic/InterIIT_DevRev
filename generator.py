from imports import *

class GenerateQuestions:
    def __init__(self, args):
        self.args = args
        file_path = os.path.join(self.args.data_dir, self.args.file_path)
        self.df = pd.read_csv(file_path)
        
        self.df =  self.df[self.df.Theme.isin(self.df.Theme.sample(5))]
        
        self.question_bank = defaultdict(list)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.generator.model_name)
        self.model = AutoModelWithLMHead.from_pretrained(self.args.generator.model_name)
        self.to(self.args.device)
        
        self.nlp = spacy.load('en_core_web_md')
        
        
    def generate_candidates(self, text):
        chunks = []
        for n in self.nlp(text).noun_chunks:
            chunk = ' '.join(token.text for token in n )
            if chunk:
                chunks.append(chunk)

        for ent in self.nlp(text).ents:
            chunks.append(ent.text)
        chunks = list(set(chunks))
        return list(set(chunks))
    
    @torch.inference_mode()
    def generate_queries(self,candidates, text, pbar=False):
        num_return_sequences = self.args.generator.num_return_sequences
        max_length_query = self.args.generator.max_length_query
        batch_size = self.args.generator.batch_size
        top_p = self.args.generator.top_p

        queries = []
        for i in tqdm.trange(0, len(candidates), batch_size, disable=not pbar):
            input_text = ["answer: %s  context: %s </s>" % (candidate, text) for candidate in candidates[i:i+batch_size]]
            features = self.tokenizer(input_text, return_tensors='pt', padding=True, 
                                      truncation=True, max_length=512).to(self.args.device)
            outputs = self.model.generate(**features, max_length=max_length_query, do_sample=True,
                                     top_p=top_p, num_return_sequences=num_return_sequences)
            queries.extend([self.tokenizer.decode(out, skip_special_tokens=True).strip('question: ') for idx, out in enumerate(outputs)])
        queries = list(set(queries))
        return queries
    
    def fit(self, pbar=True):
        for text in tqdm.tqdm(self.df.Paragraph.unique(), disable=not pbar):
            candidates = self.generate_candidates(text)
            self.question_bank[text] = self.generate_queries(candidates, text)
        self.df['GeneratedQuestions'] = self.df.Paragraph.apply(lambda x: self.question_bank[x])
        save_path = os.path.join(self.args.data_dir, self.args.generator.save_file)
        self.df.to_csv(save_path)

    def time_profile(self, questions, rankings):
        latencies = {}
        
        start = time.time()
        self.fit()
        end = time.time()
        
        latencies['generator'] = (end-start)*1000/self.df.Paragraph.nunique()
        
        return latencies
    
    def to(self, device):
        self.args.device = device
        self.model.to(device)
        