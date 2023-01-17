from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, scan

class ElasticSearchAPI:
    """
    Elastic Store python API wrapper.
    Bsic Usage:
        es = ElasticSearchAPI(                  # Connect to the server and declare object
                host='localhost',port=9200,
                index=None,index_kwargs={} 
                 )
                 
        print(len(es))                          # Check number of stored docs
        
        es[0] = {'Theme':'Testing',             # Assignment operation
                'Paragraph':'Sample para'}
                
        
        print(es[0])                            # get item by index
        
        del es[0]                               # Deletion operation
        
        es.batch_setitem(paras_json)            # Upload docs by bulk. Format: list of {id: {field:value}}
        
        query = "Who is beyonce?"
        out1 = es.search(Query)     # Query
        
        queries = ["Who is beyonce?",
        "What is natural language processing?"]
        
        out2 = es.batch_search(queries)         # Batch Query, faster
        
        out3 = es.search(query,theme='People')  # Apply theme filter
        
        out4 = es.batch_search(queries,         # Apply theme filter on bulk.
                    themes=['People',None])
                    
        
    """
    def __init__(self,host='localhost',port=9200,
                 index=None,index_kwargs={},overwrite=True):
        self.es = self.connect(host,port)
        self._create_index(index,overwrite=True,**index_kwargs)
        
        
    def connect(self,host='localhost',port=9200):
        """
        Connect to ES server
        """
        es = Elasticsearch([{'host': host, 'port': port}],timeout=10) # Increase timeout if needed
        if es.ping():
            print('Connected to server!')
            return es
        else:
            raise ConnectionError(f"Check if ES server is running and verify the port is {port}")
    

    
    def search(self,query,theme=None,top_k=10,simplify_output=True,_source=False):
        """
        BM25 similarity search with query
        """
        res = self.es.search(index=self.index,
                             query=self._construct_query(query,theme),
                             size=top_k,_source=_source)
        if simplify_output:
            res = self._simplify_output(res)
        return res
    
    
    def batch_search(self,queries,themes=None,top_k=10,simplify_output=True,batch_size=100,_source=False):
        out = []
        for i in range(0,len(queries),batch_size):
            _queries = queries[i:i+batch_size]
            _themes = None if themes is None else themes[i:i+batch_size]
            body = self._construct_bulk_queries_body(_queries,_themes,top_k,_source=_source)
            res = self.es.msearch(index=self.index,body=body)
            if simplify_output:
                res = res['responses']
                res = [self._simplify_output(e) for e in res]
            out.extend(res)
        return out
    
    def batch_upload(self,dataset):
        '''
            Data Format:
            [
                .
                .
                .
                {id:   {'key1':value1,'key2':value2...}    },
                .
                .
                .
            ]
        '''
        data_dicts = self._construct_bulk_setitem(dataset)
        out = bulk(self.es,data_dicts)
        self.refresh()
        return out
        
    def refresh(self):
        self.es.indices.refresh(index=self.index)
            
    def list_indices(self):
        return self.es.indices.get_alias(index="*")
    
    
    def delete_index(self,index):
        return self.es.indices.delete(index=index)
    
    def delete_all_indices(self):
        indices = self.list_indices().keys()
        for index in indices:
            self.delete_index(index)
            
    def create_index(self,index,settings=None,mappings=None,**kwargs):
        body = self._get_default_settings()
        return self.es.indices.create(index=index,body=body,**kwargs)

    def _simplify_output(self,out):
        hits = out['hits']['hits']
        for hit in hits:
            if '_source' in hit:
                hit.update(hit.pop('_source'))
        return hits
            
    def _create_index(self,index=None,overwrite=True,**kwargs):
        used_indices = list(self.list_indices().keys())
        if index is None:
            index = 'default'
        if index in used_indices and overwrite:
            print(f'{index} Already exists, overwriting!')
            self.delete_index(index)
        self.index = index
        self.create_index(index,**kwargs)


        
        
        
    def _construct_query(self,query_text,theme=None):
        if theme is None:
            query_dict =  {
                            "bool": {
                                "must": [
                                    {
                                        "multi_match": {
                                            "query": query_text,
                                            "type": "most_fields",
                                             "operator": 'OR',
                                                        }
                                    }
                                        ]
                                    }
                     }
        else:
             query_dict =  {
                                "bool": {
                                    "must": [
                                        {
                                            "multi_match": {
                                                "query": query_text,
                                                "type": "most_fields",
                                                 "operator": 'OR',
                                                            }
                                        },
                                        {'match': {'Theme': theme}}
                                            ]
                                        }
                         }
        
        return query_dict
    
    def _construct_body(self,query_text,theme,size=10):
        body = {"size":str(size),"query":self._construct_query(query_text,theme)}
        return body
    
    def _construct_bulk_queries_body(self,queries,themes,size=10,_source=True):
        body = []
        if themes is None:
            for query in queries:
                body.append({}) # Header
                b = self._construct_body(query,None,size)
                b['_source'] = _source
                body.append(b)
            return body
        else:
            for query, theme in zip(queries,themes):
                body.append({}) # Header
                b = self._construct_body(query_text=query,theme=theme,size=size)
                b['_source'] = _source
                body.append(b)
            return body
    
    def _construct_bulk_setitem(self,data):
        data_dicts = []
        for idx, entry in data.items():
            data_dicts.append(
                {
                    "_index": self.index,
                    "_id": idx,
                    "_source": entry
                }
                            )
        return data_dicts
    
    def __len__(self):
        count = self.es.cat.count(index=self.index, format="json")[0]['count']
        return int(count)
    
    def __getitem__(self,id):
        if id>0:
            return self.es.get(index=self.index, id=id)
        else:
            while id<0:
                id += len(self)
            return self.es.get(index=self.index, id=id)
    
    def __delitem__(self,id):
        out = self.es.delete(index=self.index, id=id)
        self.refresh()
        return out
    
    def __setitem__(self,id,document):
        out =  self.es.index(index=self.index, id=id, document=document)
        self.refresh()
        return out  
    def _get_default_settings(self): # Skipping, negligible perf. gain
        body=None
        return body