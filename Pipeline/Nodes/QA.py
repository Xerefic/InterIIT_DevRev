from transformers.utils.logging import  set_verbosity_error
set_verbosity_error()
import os
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.pipelines import pipeline as opt_pipeline
from optimum.onnxruntime import ORTModelForQuestionAnswering
from transformers import pipeline, AutoTokenizer
import pandas as pd


import os
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.pipelines import pipeline as opt_pipeline
from optimum.onnxruntime import ORTModelForQuestionAnswering
from optimum.onnxruntime import ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig

# create ORTOptimizer and define optimization configuration


# apply the optimization configuration to the model




class QA():
    def __init__(self,model=None,model_name = "deepset/tinyroberta-squad2",quantize=False,onnx=True,optimize=True,device='cpu'):
        self.qa_cuda_support = True
        if device=='cuda':device='cuda:0'
        if quantize==False and onnx==False:
            self.model = model or pipeline("question-answering", model = model_name, tokenizer = model_name,device=device)
        else:
            self.qa_cuda_support = False
            self.model = self._load_quantized(model_name,quantize,optimize)
        self.setup_flag = False
        
    def _load_quantized(self,model_name,quantize,optimize):
        file = model_name.replace('/','-')
        save_dir = f'{file}-quantized.onnx'
        files = os.listdir()
        model = ORTModelForQuestionAnswering.from_pretrained(model_name, from_transformers=True) # Load onnx model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if optimize:
            optimizer = ORTOptimizer.from_pretrained(model)
            optimization_config = OptimizationConfig(optimization_level=99) # enable all optimizations
            # apply the optimization configuration to the model
            optimizer.optimize(save_dir='temp',optimization_config=optimization_config)
            model = ORTModelForQuestionAnswering.from_pretrained('temp')
            # os.system('rm -r temp')
        
        if quantize:
            if not save_dir in files:
                print('Starting QA quantization')
                # Quantize
                quantizer = ORTQuantizer.from_pretrained(model)
                qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)
                quantizer.quantize(quantization_config=qconfig,save_dir=save_dir)
                print('QA quantized!')
            model = ORTModelForQuestionAnswering.from_pretrained(save_dir)
        model = opt_pipeline("question-answering", model=model, tokenizer=tokenizer)
        return model
            
        
    def setup(self,df):
        if self.setup_flag:
            print('Already setup is done! (QA)')
            return
        paras = df.loc[:,['Para_id','Paragraph']].drop_duplicates().set_index('Para_id',drop=True)
        self.para_lookup = dict(zip(paras.index,paras.Paragraph))
        self.setup_flag = True
        
    def __call__(self,qns,retrieved,batch=4,explore=0,exploration_cutoff=0.4):
        context = [self.para_lookup[retrd[0]] for retrd in retrieved]
        out = self.model(question=qns,context=context,handle_impossible_answer =True,batch_size=batch,max_seq_len=500)
        if isinstance(out,dict): out = [out]
        for entry,retrieves in zip(out,retrieved):
            entry['Para_id'] = retrieves[0]
            entry['depth'] = 0
        if not isinstance(out,list): out = [out]
        out_df = pd.DataFrame(out)
        
        if explore==0:return out_df
    
        reeval_idx = out_df[(out_df.score<exploration_cutoff) | (out_df.answer=='')].index
        depth = 0
        while depth<explore and len(reeval_idx):
            depth+=1
            row_ids = [idx for idx in reeval_idx if len(retrieved[idx])>depth]
            if not row_ids: return out_df
            q = [qns[idx] for idx in reeval_idx if len(retrieved[idx])>depth]
            para_ids = [retrieved[idx][depth] for idx in reeval_idx if len(retrieved[idx])>depth]
            context = [self.para_lookup[idx] for idx in para_ids]
            _out = self.model(question=q,context=context,handle_impossible_answer =True,batch_size=batch,max_seq_len=500)
            if isinstance(_out,dict):_out = [_out]
            for entry,idx in zip(_out,para_ids):
                entry['Para_id'] = idx
                entry['depth'] = depth
            out_df.iloc[row_ids] = pd.DataFrame(_out).values
            reeval_idx = out_df[(out_df.score<exploration_cutoff) | (out_df.answer=='')].index
        return out_df
    
    def to(self,device):
        if device=='cuda':device=0
        if self.qa_cuda_support:
            model = self.model.model.to('cpu')
            tokenizer = self.model.tokenizer
            self.model = pipeline("question-answering", model = model, tokenizer = tokenizer,device=device)
        else:
            print('QA onnx is on cpu!')