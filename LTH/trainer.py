from imports import *
from args import *
from data import *
from networks import *
from utils import *

class Trainer():
    def __init__(self, args):

        self.args = args
        self.start_prune_epoch = self.load()
        self.start_train_epoch = 0

        self.traindata, _, self.testdata = self.get_data()
        self.trainloader, _, _ = self.get_iterator()
        
        self.model = self.get_model()
        # self.criterion = self.get_criterion()
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()

    def get_data(self):
        traindata = Squad(self.args, mode='train')
        testdata = Squad(self.args, mode='validation')
        return traindata, None, testdata

    def get_iterator(self):
        self.traindata.infer = False
        trainloader = DataLoader(self.traindata, batch_size=self.args.batch_size, shuffle=True, worker_init_fn=np.random.seed(self.args.seed))
        # validloader = DataLoader(valid, batch_size=self.args.batch_size, shuffle=False, worker_init_fn=np.random.seed(self.args.seed))
        # testloader = DataLoader(self.testdata, batch_size=self.args.batch_size, shuffle=False, worker_init_fn=np.random.seed(self.args.seed))
        return trainloader, None, None

    def get_criterion(self):
        return nn.CrossEntropyLoss(weight=self.args.weights).to(self.args.device)
    
    def get_optimizer(self):
        # return AdamW(self.model.parameters(), lr=self.args.lr)
        return torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)

    def get_scheduler(self):
        return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.args.max_train_epochs, eta_min=1e-12, last_epoch=-1, verbose=False)

    def get_model(self):
        model = Model(self.args)
        return model.to(self.args.device)

    def get_model_params(self):
        return sum(p.numel() for p in self.model.parameters())/1e6

    def save(self, epoch, best=False):
        if best:
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                }, os.path.join(self.args.checkpoint, f"model_best_{self.args.experiment}.pth"))
        else: 
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                }, os.path.join(self.args.checkpoint, f"model_epoch{epoch}_{self.args.experiment}.pth"))
        
    def load(self):
        if os.path.exists(os.path.join(self.args.checkpoint, f"model_{self.args.experiment}.pth")):
            checkpoints = torch.load(os.path.join(self.args.checkpoint, f"model_{args.experiment}.pth"), map_location=self.args.device)
            self.model.load_state_dict(checkpoints['model_state_dict'])
            self.optimizer.load_state_dict( checkpoints['optimizer_state_dict'])
            return checkpoints['epoch']
        return 0

    def prune(self):
        init = copy.deepcopy(self.model.state_dict())

        if self.args.prune_type=="random":
            random_prune(self.model, self.args.prune_rate)
        elif self.args.prune_type=="l1":
            l1_prune_custom(self.model, self.args.prune_rate, self.args.prune_ff_only)

        curr_mask = extract_mask(self.model.state_dict())

        if self.args.prune_type=="random":
            remove_prune(self.model)
        elif self.args.prune_type=="l1":
            remove_prune_custom(self.model, self.args.prune_ff_only)
        self.model.load_state_dict(init, strict=False)

        if self.args.prune_type=="random":
            mask_prune(self.model, curr_mask)
        elif self.args.prune_type=="l1":
            mask_prune_custom(self.model, curr_mask, self.args.prune_ff_only)
        
        self.start_train_epoch = 0
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()
        sparsity = check_sparsity_custom(self.model, prune_ff_only=self.args.prune_ff_only)
        return sparsity

    def train(self):
        epoch_loss = 0
        torch.cuda.empty_cache()
        self.model.train()

        with torch.autograd.set_detect_anomaly(True):
            bar = pyprind.ProgBar(len(self.trainloader), bar_char='█')
            for index, (input_ids, attention_mask, start_positions, end_positions) in enumerate(self.trainloader):
                input_ids, attention_mask, start_positions, end_positions = input_ids.to(self.args.device), attention_mask.to(self.args.device), start_positions.to(self.args.device), end_positions.to(self.args.device)
                self.optimizer.zero_grad()
                
                output = self.model(input_ids, attention_mask, start_positions, end_positions)

                loss = output[0]
                loss.backward()

                self.optimizer.step()

                epoch_loss += loss.item()/len(self.trainloader)

                bar.update()
                torch.cuda.empty_cache()

        return epoch_loss

    def evaluate(self):
        epoch_metric = 0
        self.testdata.infer = True
        self.model.eval()
        for index in tqdm.trange(len(self.testdata)):
            input, answer = self.testdata[index]
            for key in input.keys():
                input[key] = input[key].to(self.args.device)
            epoch_metric += self.model.score(input, answer)['f1']/len(self.testdata)
        return epoch_metric
    
    def fit(self, next=True):
        if next:
            self.start_epoch = self.load()
        best_accuracy = 0
        for epoch_prune in range(self.start_prune_epoch+1, self.args.max_prune_epochs+1, 1):
            
            for epoch_train in range(self.start_train_epoch+1, self.args.max_train_epochs+1, 1):
                epoch_train_loss = self.train()
                epoch_test_metric = self.evaluate()

                if best_accuracy < epoch_test_accuracy:
                    best_accuracy = epoch_test_accuracy
                    self.save(None, best=True)

            epoch_sparsity = self.prune()
            self.save(epoch_prune, best=False)

            self.scheduler.step()
            time.sleep(1)
            print(f'Epoch {epoch_prune}/{self.args.max_prune_epochs} | Training: Loss = {round(epoch_train_loss, 4)}  Sparsity = {epoch_sparsity} | Testing: F1 = {round(epoch_test_metric, 4)}')

if __name__ == "__main__":
    args = TrainingArgs()
    trainer = Trainer(args)