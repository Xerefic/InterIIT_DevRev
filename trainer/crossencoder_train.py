from imports import *

class CrossEncoderCustom(CrossEncoder):
    def fit(self,
            train_dataloader: DataLoader,
            evaluator: SentenceEvaluator = None,
            epochs: int = 1,
            loss_fct = None,
            activation_fct = nn.Identity(),
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = torch.optim.AdamW,
            optimizer_params: Dict[str, object] = {'lr': 2e-5},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
            show_progress_bar: bool = True,
            use_teacher: bool = False,
            distill_logits: bool = False,
            temperature: int = 6,
            alpha: float = 0.4,
            m: float = 0.999,
            ema_when: str = 'epoch',
            ):
        
        train_dataloader.collate_fn = self.smart_batching_collate

        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.model.to(self._target_device)

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)

        self.best_score = -9999999
        num_train_steps = int(len(train_dataloader) * epochs)

        # Prepare optimizers
        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        if isinstance(scheduler, str):
            scheduler = SentenceTransformer._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

        if loss_fct is None:
            loss_fct = nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()
        loss_kl = nn.KLDivLoss()    
        

        teacher = copy.deepcopy(self.model)
        teacher.eval()
        
        skip_scheduler = False
        for epoch in tqdm.trange(epochs, desc="Epoch", disable=True):
            training_steps = 0
            self.model.zero_grad()
            self.model.train()
            

            for features, labels in tqdm.tqdm(train_dataloader, desc="Iteration", smoothing=0.05, disable=True):
                if use_amp:
                    with autocast():
                        model_predictions = self.model(**features, return_dict=True)
                        logits = activation_fct(model_predictions.logits)
                        if self.config.num_labels == 1:
                            logits = logits.view(-1)
                        loss_value = loss_fct(logits, labels)
                        
                    if use_teacher:
                        teacher_predictions = teacher(**features, return_dict=True)
                        teacher_logits = activation_fct(teacher_predictions.logits)
                        loss_diff = (teacher_logits - logits).pow(2).mean()
                        loss_value = loss_value + loss_diff
                        if distill_logits:
                            loss_kl = kl_criterion(F.log_softmax(student_logits/temperature, dim=1),
                                                        F.softmax(teacher_logits/temperature, dim=1) * (temperature * temperature))
                            loss_value = loss_value * (1 - alpha) + loss_diff + loss_kl * alpha

                    scale_before_step = scaler.get_scale()
                    scaler.scale(loss_value).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    skip_scheduler = scaler.get_scale() != scale_before_step
                else:
                    model_predictions = self.model(**features, return_dict=True)
                    logits = activation_fct(model_predictions.logits)
                    if self.config.num_labels == 1:
                        logits = logits.view(-1)
                    loss_value = loss_fct(logits, labels)
                    
                    if use_teacher:
                        teacher_predictions = teacher(**features, return_dict=True)
                        teacher_logits = activation_fct(teacher_predictions.logits)
                        loss_diff = (teacher_logits - logits).pow(2).mean()
                        loss_value = loss_value + loss_diff
                        if distill_logits:
                            loss_kl = kl_criterion(F.log_softmax(student_logits/temperature, dim=1),
                                                        F.softmax(teacher_logits/temperature, dim=1) * (temperature * temperature))
                            loss_value = loss_value * (1 - alpha) + loss_diff + loss_kl * alpha
                    
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()

                optimizer.zero_grad()

                if not skip_scheduler:
                    scheduler.step()
                
                if ema_when=='step' and use_teacher:
                    for t_param, s_param in zip(teacher.parameters(), self.model.parameters()):
                        t_param.data.copy_(t_param.data * m + s_param.data * (1-m))

                training_steps += 1

                if evaluator is not None and evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps, callback)

                    self.model.zero_grad()
                    self.model.train()
            
            if ema_when=='epoch' and use_teacher:
                for t_param, s_param in zip(teacher.parameters(), self.model.parameters()):
                    t_param.data.copy_(t_param.data * m + s_param.data * (1-m))

            if evaluator is not None:
                self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1, callback)

class ReRankerTrainer():
    def __init__(self, args, df, pipe):
        self.args = args
        self.base_model = CrossEncoderCustom(self.args.themetrainer.ranker_model_name)
        self.to(self.args.device)
        self.data = None
        
    def to(self, device):
        self.args.device = device
        self.base_model._target_device = device
                    
    def collect_data(self, df, pipe, n_questions):
        data = []
        para = copy.deepcopy(df)
        para = para.loc[:, ['Para_id', 'Paragraph', 'Theme', 'GeneratedQuestions']]
        para = para.drop_duplicates(subset=['Para_id', 'Paragraph', 'Theme'])
        for _, row in tqdm.tqdm(para.iterrows(), total=len(para)):
            qns = row.GeneratedQuestions
            if len(row.GeneratedQuestions) > n_questions:
                qns = random.sample(row.GeneratedQuestions,n_questions)
            retrieved,_ = pipe.retriever.predict(qns,row.Theme)
            for i, qn in enumerate(qns):
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
            if len(retrieved) > n_negatives:
                negative_ids = random.sample(retrieved, n_negatives)
            else:
                negative_ids = retrieved
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
        if self.data is None:
            self.data = self.collect_data(df, pipe, self.args.themetrainer.n_questions)
        para_lookup = dict(zip(df.Para_id, df.Paragraph))
        for theme, G in tqdm.tqdm(self.data.groupby('Theme'), total=self.data.Theme.nunique()):
            examples = self.create_examples(G.to_dict('records'), para_lookup, self.args.themetrainer.n_negatives)
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
                      output_path=None,
                      use_teacher=self.args.themetrainer.use_teacher,
                      distill_logits=self.args.themetrainer.distill_logits,
                      ema_when=self.args.themetrainer.ema_when,
                     )
            score = evaluator(model)
            print(f'Score After = {score}', end='\n\n')
            theme = re.sub("\.", "", theme)
            path = os.path.join(self.args.saves_path, f'crossencoder_{theme}.onnx')
            self.save_onnx(model, path)
