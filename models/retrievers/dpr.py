from imports import *
from utils import clean_text

class DenseRetriever():
    def __init__(self, args, df):
        self.args = args
        self.backend = None
        if self.args.device=='cpu':
            self.load_onnx()
        elif self.args.device=='cuda':
            self.load_torch()
        
        self.fit(df)
    
    def fit(self, df):
        batch_size = self.args.retriever.dpr_batch_size
        model_P = SentenceTransformer(self.args.retriever.dpr_model_P).to(self.args.device)
        cols = ['Para_id', 'Paragraph', 'Theme']
        assert df.columns.isin(cols).sum()==3 , "Verify Para_id, Paragraph, Theme are in the columns"
        
        paras = df.loc[:,cols].drop_duplicates().reset_index(drop=True)
        self.idx_to_para_id = dict(zip(paras.index, paras.Para_id))
        paras.loc[:,'Paragraph'] = paras.Paragraph.apply(lambda x: clean_text(x, stem=False))
        self.embeddings = model_P.encode(paras.Paragraph, batch_size=batch_size, show_progress_bar=True)
        self.theme_to_idxs = paras.index.to_series().groupby(paras.Theme).unique()
        
    def predict(self, questions, theme):
        batch_size = self.args.retriever.dpr_batch_size
        questions = [clean_text(q, stem=False) for q in questions]
        dense_qns = self.model_Q.encode(questions, batch_size=batch_size, show_progress_bar=False)
        idx = self.theme_to_idxs[theme].tolist()
        top_k = self.args.retriever.dpr_top_k
        
        embeds = self.embeddings[idx]
        rankings = semantic_search(dense_qns,embeds,top_k=top_k)
        scores = [ [ rank['score']  for rank in ranks] for ranks in rankings ]
        rankings = [ [ self.idx_to_para_id[idx[rank['corpus_id']]]  for rank in ranks] for ranks in rankings ]
        return rankings, scores
    
    def to(self, device):
        if self.backend=='torch':
            self.args.device = device
            self.model_Q.to(device)
            self.model_Q._target_device = device
        
    def load_torch(self):
        if self.backend!='torch':
            self.backend = 'torch'
            self.model_Q = SentenceTransformer(self.args.retriever.dpr_model_Q)
            self.to(self.args.device)
            self.model_Q.eval()
            
    def load_onnx(self):
        if self.backend!='onnx':
            self.backend = 'onnx'
            self.model_Q = sentence_transformers_onnx(self.model_Q, 'temp')
            os.system('rm -rf temp')
            os.system('rm temp.onnx')
            
            
            
class OnnxEncoder:
    """
    Credits: https://github.com/UKPLab/sentence-transformers/issues/46
    OnnxEncoder dedicated to run SentenceTransformer under OnnxRuntime.
    """

    def __init__(self, session, tokenizer, pooling, normalization):
        self.session = session
        self.tokenizer = tokenizer
        self.max_length = tokenizer.__dict__["model_max_length"]
        self.pooling = pooling
        self.normalization = normalization

    def encode(self, sentences: list, **kwargs):

        sentences = [sentences] if isinstance(sentences, str) else sentences

        inputs = {
            k: v.numpy()
            for k, v in self.tokenizer(
                sentences,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).items()
        }

        hidden_state = self.session.run(None, inputs)
        sentence_embedding = self.pooling.forward(
            features={ "token_embeddings": torch.Tensor(hidden_state[0]), "attention_mask": torch.Tensor(inputs.get("attention_mask")),},
        )

        if self.normalization is not None:
            sentence_embedding = self.normalization.forward(features=sentence_embedding)

        sentence_embedding = sentence_embedding["sentence_embedding"]

        if sentence_embedding.shape[0] == 1:
            sentence_embedding = sentence_embedding[0]

        return sentence_embedding.numpy()
    
def sentence_transformers_onnx(model, path, do_lower_case=True, input_names=["input_ids", "attention_mask", "segment_ids"],
        providers=["CPUExecutionProvider"]):

        model.save(path)
        configuration = AutoConfig.from_pretrained(path, from_tf=False, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(path, do_lower_case=do_lower_case, from_tf=False, local_files_only=True)
        encoder = AutoModel.from_pretrained(path, from_tf=False, config=configuration, local_files_only=True)

        st = ["dummy"]

        inputs = tokenizer(st, padding=True, truncation=True, max_length=tokenizer.__dict__["model_max_length"], return_tensors="pt",)
        
        model.eval()
        with torch.no_grad():

            symbolic_names = {0: "batch_size", 1: "max_seq_len"}

            torch.onnx.export(encoder, args=tuple(inputs.values()),
                f=f"{path}.onnx", opset_version=13,  # ONX version needs to be >= 13 for sentence transformers.
                do_constant_folding=True, input_names=input_names, output_names=["start", "end"],
                dynamic_axes={
                    "input_ids": symbolic_names,
                    "attention_mask": symbolic_names,
                    "segment_ids": symbolic_names,
                    "start": symbolic_names,
                    "end": symbolic_names,
                },
            )

            normalization = None
            for modules in model.modules():
                for idx, module in enumerate(modules):
                    if idx == 1:
                        pooling = module
                    if idx == 2:
                        normalization = module
                break

            return OnnxEncoder(session=onnxruntime.InferenceSession(f"{path}.onnx", providers=providers),
                tokenizer=tokenizer, pooling=pooling, normalization=normalization,)