from imports import *

# from ._colbert import ColBERT

class ColBERTConfig(PretrainedConfig):
    model_type = "ColBERT"
    bert_model: str
    compression_dim: int = 768
    dropout: float = 0.0
    return_vecs: bool = False
    trainable: bool = True
    
class ColBERT(PreTrainedModel):
    """
    ColBERT model from: https://arxiv.org/pdf/2004.12832.pdf
    """
    config_class = ColBERTConfig
    base_model_prefix = "bert_model"

    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.bert_model = AutoModel.from_pretrained(cfg.bert_model)

        for p in self.bert_model.parameters():
            p.requires_grad = cfg.trainable

        self.compressor = torch.nn.Linear(self.bert_model.config.hidden_size, cfg.compression_dim)

    def forward(self, query, document):
        query_vecs = self.forward_representation(query)
        document_vecs = self.forward_representation(document)
        score = self.forward_aggregation(query_vecs,document_vecs, query["attention_mask"], document["attention_mask"])
        return score

    def forward_representation(self, tokens, sequence_type=None):
        vecs = self.bert_model(**tokens)[0]
        vecs = self.compressor(vecs)
        if sequence_type == "doc_encode" or sequence_type == "query_encode": 
            vecs = vecs * tokens["tokens"]["mask"].unsqueeze(-1)
        return vecs

    def forward_aggregation(self, query_vecs, document_vecs, query_mask, document_mask, **kwargs):
        score = torch.bmm(query_vecs, document_vecs.transpose(2,1))
        exp_mask = document_mask.bool().unsqueeze(1).expand(-1, score.shape[1],-1)
        score[~exp_mask] = - 10000
        score = score.max(-1).values
        score[~(query_mask.bool())] = 0
        score = score.sum(-1)
        return score
    
    

class ColBertRetriever():
    def __init__(self, args, df):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased") 
        self.model = ColBERT.from_pretrained("sebastian-hofstaetter/colbert-distilbert-margin_mse-T2-msmarco").to(self.args.device)
        self.model.eval()
        
        self.fit(df)
    
    def fit(self, df):
        df = copy.deepcopy(df)
        self.para_embeds = {}
        for theme, G in tqdm.tqdm(df.groupby('Theme'), total=df.Theme.nunique()):
            paras = G.Paragraph.unique().tolist()
            para_ids = G.Para_id.unique().tolist()
            self.para_embeds[theme] = self.encode_paras(paras,para_ids)
        
    @torch.inference_mode()
    def encode_paras(self, paras, para_ids):
        batch_size = self.args.retriever.colbert_batch_size
        para_embeds = []
        masks = []
        for i in range(0, len(paras), batch_size):
            batch = self.tokenizer(paras[i:i+batch_size], return_tensors='pt', padding='max_length',
                                  truncation=True, max_length=self.args.retriever.colbert_para_maxlength).to(self.args.device)
            para_embeds.append(self.model.forward_representation(batch).cpu())
            masks.append(batch['attention_mask'].cpu())
        para_embeds = torch.concat(para_embeds, dim=0)
        masks = torch.concat(masks, dim=0)
        return {'document_vecs': para_embeds,'document_mask': masks,'para_ids': para_ids}

    @torch.inference_mode()
    def encode_query(self, qs):
        batch_size = self.args.retriever.colbert_batch_size
        qs_embeds = []
        masks = []
        for i in range(0,len(qs),batch_size):
            batch = self.tokenizer(qs[i:i+batch_size], return_tensors='pt', padding=True, 
                                   truncation=True, max_length=40).to(self.args.device)
            qs_embeds.extend(self.model.forward_representation(batch).cpu())
            masks.extend(batch['attention_mask'].cpu())
        qs_embeds = [q.unsqueeze(0) for q in qs_embeds]
        masks = [m.unsqueeze(0) for m in masks]
        return qs_embeds, masks

    def predict(self, questions, theme):
        rankings = []
        fin_scores = []
        qs_embeds, qs_masks = self.encode_query(questions)
        top_k = self.args.retriever.colbert_top_k
        
        doc_vecs = self.para_embeds[theme]['document_vecs'].to(self.model.device)
        doc_masks = self.para_embeds[theme]['document_mask'].to(self.model.device)
        n = len(doc_vecs)
        
        for i in range(len(questions)):
            q, m = torch.concat([qs_embeds[i]]*n,0), torch.concat([qs_masks[i]]*n,0)
            q = q.to(self.args.device)
            m = m.to(self.args.device)
            
            scores = self.model.forward_aggregation(query_vecs=q, query_mask=m, document_vecs = doc_vecs, document_mask = doc_masks).detach().cpu().numpy()
            ids = copy.deepcopy(self.para_embeds[theme]['para_ids'])
            scores_map = dict(zip(ids, scores))
            ids.sort(key=lambda x: scores_map[x], reverse=True)
            rankings.append(ids[:top_k])
            fin_scores.append([scores_map[idx] for idx in ids[:top_k]])
        return rankings, fin_scores
    
    def to(self, device):
        self.model.to(device)

