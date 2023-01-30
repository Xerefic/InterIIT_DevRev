from imports import *
# to do handling Question generation for BM25
# optimization
# ranker - crossencoder + kimcnn


@dataclass
class Retriever:
    ensemble = ['bm25', 'dpr', 'colbert']
    
    k1: float = 3
    b: float = 0.75
    sparse_top_k: int = 100
    
    dpr_model_Q: str = 'flax-sentence-embeddings/multi-QA_v1-mpnet-asymmetric-Q'
    dpr_model_P: str = 'flax-sentence-embeddings/multi-QA_v1-mpnet-asymmetric-A'
    dpr_top_k: int = 100
    
    colbert_batch_size: int = 4
    colbert_para_maxlength: int = 300 # If increased will cost more memory
    colbert_qn_maxlength: int = 40 # If increased, increases performance with higher latency
    colbert_top_k: int = 100
    
    weights = (['bm25', 0.33], ['dpr', 0.33, ], ['colbert', 0.33])
    voting_top_k: int = 5
    
@dataclass
class Reader:
    onnx: bool = False  # onnx cpu only
    model_name: str = "deepset/tinyroberta-squad2"
    batch_size: int = 4
    max_seq_len: int = 400
    

@dataclass
class TrainingArgs:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    retriever = Retriever()
    reader = Reader()
    
    root_dir: str = './'
    checkpoints_dir: str = f'{root_dir}/checkpoints/'
    data_dir: str = f'{root_dir}/data/'
    file_path: str = 'testset_A.csv'
    joblib_path: str = 'pipeline.joblib'