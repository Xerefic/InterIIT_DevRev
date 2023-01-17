from imports import *
from args import *

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(args.architecture)
        self.model = AutoModelForQuestionAnswering.from_pretrained(args.architecture)

    def forward(self, input_ids, attention_mask, start_positions=None, end_positions=None):
        if start_positions is None and end_positions is None:
            return self.model(input_ids, attention_mask=attention_mask)
        else:
            return self.model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)

    def decode(self, input):
        output = self(input_ids=input['input_ids'], attention_mask=input['attention_mask'])
        start_index = output.start_logits.argmax()
        end_index = output.end_logits.argmax()
        answer_tokens = input.input_ids[0, start_index:end_index+1]
        return self.tokenizer.decode(answer_tokens)

if __name__ == "__main__":
    args = TrainingArgs()
    model = Model(args)
    print(model)