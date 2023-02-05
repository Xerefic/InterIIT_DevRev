#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pipeline import Pipeline
from args import TrainingArgs
from imports import *
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('data/testset_A_with_qns.csv')


# In[3]:


df['GeneratedQuestions'] = df.GeneratedQuestions.apply(literal_eval)


# In[4]:


args = TrainingArgs()
# pipe = Pipeline(args)
pipe = Pipeline.load_from_checkpoint(args,device='cuda')
pipe.retriever.args.retriever.use_ranker = False


# In[5]:


pipe.warmup()


# In[6]:


# from sentence_transformers.evaluation import BinaryClassificationEvaluator
# train_batch_size = 8
# train_dataloader = DataLoader(train, shuffle=True, batch_size=train_batch_size)
# model_name = 'cross-encoder/ms-marco-MiniLM-L-4-v2'
# text1 = [example.texts[0] for example in test]
# text2 = [example.texts[1] for example in test]
# label = [example.label for example in test]
# evaluator = BinaryClassificationEvaluator(text1,text2,label)


# In[ ]:





# In[7]:


@dataclass
class CrossEncoderTrainingArgs:
    train_batch_size: int = 16
    model_name: str = 'cross-encoder/ms-marco-MiniLM-L-4-v2'
    num_epochs:int = 1
    train_frac:float = 0.8
    save_path: str = 'saves/model.pt'
    n_negs:int = 2
    n_questions_per_para: int = 100


# In[8]:


from sentence_transformers import InputExample
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator

class Trainer():
    def __init__(self,args,df,pipe):
        self.args = args
        self.base_model = CrossEncoder(args.model_name)
        self.fit(df,pipe)
                    
    def collect_data(self,df,pipe,n_questions = 10):
        data = []
        for _,row in tqdm.tqdm(df.iterrows(),total=len(df)):
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
    
    def create_examples(self,data,para_lookup):
        n_negs = self.args.n_negs
        examples = []
        for entry in tqdm.tqdm(data,disable=True):
            examples.append(
            InputExample(
                texts=[entry['Question'] , entry['Paragraph']], label=1
            ))
            retr = entry['retrieved'].copy()
            if entry['Para_id'] in retr:
                retr.remove(entry['Para_id'])
            neg_ids = random.sample(entry['retrieved'],2)
            for idx in neg_ids:
                examples.append(
                        InputExample(
                            texts=[entry['Question'] , para_lookup[idx]],label= 0
                        ))
        return examples
    
    def save_onnx(self,model, path, do_lower_case=True, input_names=["input_ids", "attention_mask", "token_type_ids"]):
        providers = [args.onnx_provider]
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
            
    def fit(self,df,pipe):
        data = self.collect_data(df,pipe,self.args.n_questions_per_para)
        para_lookup = dict(zip(df.Para_id,df.Paragraph))
        for theme,G in tqdm.tqdm(data.groupby('Theme'),total=data.Theme.nunique()):
            examples = self.create_examples(G.to_dict('records'),para_lookup)
            n = int(len(examples) * self.args.train_frac)
            train = examples[:n]
            test = examples[n:]
            evaluator = CEBinaryClassificationEvaluator.from_input_examples(test, name='Testing')
            print(theme)
            print('Training data points: ',len(train))
            train_dataloader = DataLoader(train,shuffle=True, batch_size=self.args.train_batch_size)
            warmup_steps = int(len(train_dataloader) * self.args.num_epochs * 0.1) #10% of train data for warm-up
            model = copy.deepcopy(self.base_model)
            score = evaluator(model)
            print('Score before: ',score)
            model.fit(train_dataloader=train_dataloader,
                      epochs=self.args.num_epochs,
                      warmup_steps=warmup_steps,
                      output_path=None)
            score = evaluator(model)
            print('Score after: ',score,'\n\n')
            theme = re.sub("\.", "", theme)
            path = os.path.join('saves', f'crossencoder_{theme}.onnx')
            self.save_onnx(model,path)


# In[68]:


test_df = df[df.Theme.isin(df.Theme.unique()[:10])]


# In[69]:


training_args = CrossEncoderTrainingArgs()
training_args.n_questions_per_para = 50
trainer = Trainer(training_args,test_df,pipe)


# In[12]:


pipe.retriever.args.retriever.reranker = False
pipe.retriever.ranker.backend = 'onnx' # temp


# In[13]:


def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
  if not s: return []
  return normalize_answer(s).split()

def calc_f1(a_gold, a_pred):
  gold_toks = get_tokens(a_gold)
  pred_toks = get_tokens(a_pred)
  common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
  num_same = sum(common.values())
  if len(gold_toks) == 0 or len(pred_toks) == 0:
    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
    return int(gold_toks == pred_toks)
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(pred_toks)
  recall = 1.0 * num_same / len(gold_toks)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1

def calc_max_f1(predicted, ground_truths):
  max_f1 = 0
  if len(ground_truths) == 0:
    return len(predicted) == 0
  for ground_truth in ground_truths:
    f1 = calc_f1(predicted, ground_truth)
    max_f1 = max(max_f1, f1)
  return max_f1


# In[61]:


def evaluate_themes(pipeline, df):
    top1_old = {}
    top1 = {}
    f1 = {}
    for theme, G in df.groupby('Theme'):
        questions = G.Question.tolist()
        para_ids = G.Para_id.tolist()
        answers = G.Answer_text.apply(lambda x: literal_eval(x))
        rankings, scores = pipeline.retriever.predict(questions, theme)
        scores_ = []
        for r, p in zip(rankings, para_ids):
            scores_.append(int(r[0]==p))
        top1_old[theme] = np.mean(scores_)
        
        pipe.retriever.ranker.load_theme_model(theme)
        
        to_pred_idx = []
        cutoff = 0.1
        for i,score in enumerate(scores):
            diff = score[0]-score[1]
            if diff < cutoff: 
                to_pred_idx.append(i)
        _rankings = [rankings[i] for i in to_pred_idx]
        _questions = [questions[i] for i in to_pred_idx]
        _rankings,_scores = pipe.retriever.ranker.predict(_questions, _rankings)
        for i,idx in enumerate(to_pred_idx):
            rankings[idx] = _rankings[i]
            scores[idx] = _scores[i]
        
        # rankings, scores = pipe.retriever.ranker.predict(questions,rankings)
        # print(rankings,scores)
        
        predicted = [p['answer'] for p in pipeline.reader.predict(questions, rankings)]  
        
        scores = []
        for r, p in zip(rankings, para_ids):
            scores.append(int(r[0]==p))
        top1[theme] = np.mean(scores)
        
        scores = []
        for p, a in zip(predicted, answers):
            scores.append(calc_max_f1(p, a))
        f1[theme] = np.mean(scores)
        print(f'Theme: {theme} | Top1 = {round(top1[theme], 4)} , F1 = {round(f1[theme], 4)} , Top1_no_ranker = {round(top1_old[theme], 4)}')
    return top1,top1_old, f1


# In[62]:


themes = df.Theme.unique()[:10]
top1,top1_old, f1 = evaluate_themes(pipe,df[df.Theme.isin(themes)])


# In[67]:


top1_, f1_ ,top1_old_ = 0, 0, 0
scores = {}
for k in top1.keys():
    top1_ += top1[k]/len(top1.keys())
    top1_old_ += top1_old[k]/len(top1.keys())
    f1_ += f1[k]/len(f1.keys())   
    scores[k] = {'top1': top1[k], 'top1': top1_old[k],'f1': f1[k]}

with open('checkpoints/scores_ranker_electra.json', 'w') as f:
    json.dump(scores, f)
        
print(f'No Ranker | Top1 = {round(top1_, 4)} Top1_no_ranker = {round(top1_old_, 4)} , F1 = {round(f1_, 4)}')


# In[8]:


# for theme in themes:
#     G = df[df.Theme==theme]
#     qns = G.Question.tolist()
#     rankings,scores = pipe.retriever.predict(qns,theme)
#     break


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[46]:


n_qns = 10
data = []
for theme, G in tqdm.tqdm(df.groupby('Theme'),total=df.Theme.nunique()):
    for _,row in G.iterrows():
        samples = random.sample(row.GeneratedQuestions,n_qns)
        out,scores = pipe.retriever.predict(samples,theme)
        for o,s,qn in zip(out,scores,samples):
            entry = row.to_dict()
            entry.update({
                'GeneratedQuestion': qn,
                'retr': o
            })
            data.append(entry)


# In[47]:


data_df = pd.DataFrame(data)


# In[48]:


data_df.shape


# In[49]:


data_df.to_csv('gen_qns_retr.csv',index=False)


# In[50]:


get_ipython().system('wormhole send gen_qns_retr.csv')


# In[ ]:




