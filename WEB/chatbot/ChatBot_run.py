import pandas as pd
import tqdm     # 반복문 진행률

import torch
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

def chatanswer(request):
    Q_TKN = '<usr>'
    A_TKN = '<sys>'
    BOS = '</s>'
    EOS = '</s>'
    MASK = '<unused0>'
    PAD = '<pad>'
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

    koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                bos_token=BOS, eos_token=EOS, unk_token='<unk>',
                pad_token=PAD, mask_token=MASK)
    model = torch.load('WEB\static\model_save_v10.pt', map_location = device)

    a = ''
    with torch.no_grad():
        while 1:
            input_ids = torch.LongTensor(koGPT2_TOKENIZER.encode(Q_TKN + request + A_TKN + a)).unsqueeze(dim=0).to(device)
            pred = model(input_ids)
            pred = pred.logits
            gen = koGPT2_TOKENIZER.convert_ids_to_tokens(torch.argmax(pred.to('cpu'), dim=-1).squeeze().numpy().tolist())[-1]
            if gen == EOS:
                break
            a += gen.replace('▁', " ")
        print(a.strip())

q =  input('user > ')
chatanswer(q)