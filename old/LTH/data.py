from imports import *
from args import *

class Squad(Dataset):
    def __init__(self, args, mode='train'):
        self.infer = False
        self.data = load_dataset("squad_v2", split=mode)
        self.tokenizer = AutoTokenizer.from_pretrained(args.architecture)

    def add_end_idx(self, answer, context):
        gold_text = answer['text'][0]
        start_idx = answer['answer_start'][0]
        end_idx = start_idx + len(gold_text)
        if context[start_idx:end_idx] == gold_text:
            answer['answer_start'] = start_idx
            answer['answer_end'] = end_idx
        elif context[start_idx-1:end_idx-1] == gold_text:
            answer['answer_start'] = start_idx
            answer['answer_end'] = end_idx-1
        elif context[start_idx-2:end_idx-2] == gold_text:
            answer['answer_start'] = start_idx-2
            answer['answer_end'] = end_idx-2
        return answer, context

    def __getitem__(self, index):
        data = self.data[index]
        question = data['question']
        if len(data['answers']['text'])>0:
            answer, context = self.add_end_idx(data['answers'],  data['context'])
            encoding = self.tokenizer(question, context, return_tensors="pt", truncation=True, padding='max_length')
            encoding.update({'start_positions': torch.tensor([answer['answer_start']]).unsqueeze(0), 'end_positions': torch.tensor([answer['answer_end']-1]).unsqueeze(0)})
            answer['text'] = answer['text'][0]
        else:
            index = np.random.randint(len(self))
            return self[index]
        if not self.infer:
            return encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0), encoding['start_positions'].squeeze(0), encoding['end_positions'].squeeze(0)
        else:
            references = {'answers': {'answer_start': [answer['answer_start']], 'text': [answer['text']]}, 'id': data['id']}            
            return encoding, references

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    args = TrainingArgs()
    dataset = Squad(args)
    print(len(dataset))
