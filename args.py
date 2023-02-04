from imports import *
# optimization
# ranker - crossencoder + kimcnn


@dataclass
class RetrieverArgs:
    ensemble = ['bm25', 'dpr', 'colbert']
    
    k1: float = 3
    b: float = 0.75
    sparse_top_k: int = 300 
    
    dpr_batch_size: int = 32
    dpr_model_Q: str = 'flax-sentence-embeddings/multi-QA_v1-mpnet-asymmetric-Q'
    dpr_model_P: str = 'flax-sentence-embeddings/multi-QA_v1-mpnet-asymmetric-A'
    dpr_top_k: int = 300
    
    colbert_batch_size: int = 32
    colbert_model_name: str = "sebastian-hofstaetter/colbert-distilbert-margin_mse-T2-msmarco"
    colbert_para_maxlength: int = 300
    colbert_qn_maxlength: int = 40
    colbert_top_k: int = 300
    
    weights: tuple = (['bm25', 0.3], ['dpr', 0.2], ['colbert', 0.5])
    voting_top_k: int = 5
    
    ranker_treshold_path: str = 'coefficients.json'
    use_ranker: bool = False
    reranker: str = 'cross-encoder'
    ranker_model_name: str = 'cross-encoder/ms-marco-MiniLM-L-4-v2'
    ranker_batch_size: int = 8
    
@dataclass
class ReaderArgs:
    onnx: bool = False  # onnx cpu only
    model_name: str = "bhadresh-savani/electra-base-squad2" #"deepset/tinyroberta-squad2"
    batch_size: int = 32
    max_seq_len: int = 400
    
@dataclass
class GeneratorArgs:
    model_name: str = "mrm8488/t5-base-finetuned-question-generation-ap"
    batch_size: int = 64
    top_p: float = 0.98
    num_return_sequences: int = 2
    max_length_query: int = 64
    file_path: str = 'testset_A.csv'
    save_file: str = 'test_set_A_qn_generated.csv'
    

@dataclass
class TrainingArgs:
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_jobs: int = os.cpu_count()
    onnx_provider: str = "CPUExecutionProvider"
    onnx_float16: bool = False
    
    generator = GeneratorArgs()
    retriever = RetrieverArgs()
    reader = ReaderArgs()
    
    root_dir: str = './'
    checkpoints_dir: str = f'{root_dir}/checkpoints/'
    data_dir: str = f'{root_dir}/data/'
    file_path: str = 'testset_A_with_qns.csv'
    saves_path: str = f'{root_dir}/saves/'
    joblib_path: str = 'pipeline.joblib'