from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import tqdm     # 반복문 진행률


import torch
import warnings
warnings.filterwarnings('ignore')
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel



# Create your views here.
def index(request):
    context = {}
    return render(request, 'index.html', context)

def chathome(request):
    context = {}
    return render(request, 'chat.html', context)
    

@csrf_exempt
def chatanswer(request):
    context = {}
    chattext = request.GET.get('ctext')
   
    context['result'] = chattext

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
    model = torch.load('static\\model_save_v10.pt', map_location = device)

    a = ''
    with torch.no_grad():
        while 1:
            input_ids = torch.LongTensor(koGPT2_TOKENIZER.encode(Q_TKN + chattext + A_TKN + a)).unsqueeze(dim=0).to(device)
            pred = model(input_ids)
            pred = pred.logits
            gen = koGPT2_TOKENIZER.convert_ids_to_tokens(torch.argmax(pred.to('cpu'), dim=-1).squeeze().numpy().tolist())[-1]
            if gen == EOS:
                break
            a += gen.replace('▁', " ")
        
    context['anstext'] = a
    return JsonResponse(context, content_type = 'application/json')

# q =  input('user > ')
# chatanswer(q)
