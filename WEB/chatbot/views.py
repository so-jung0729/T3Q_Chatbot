from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import tqdm     # 반복문 진행률

import tensorflow as tf
from transformers import AutoTokenizer
from transformers import TFGPT2LMHeadModel
import torch

import warnings
warnings.filterwarnings('ignore')



# Create your views here.
def home(request):
    context = {}
    return render(request, 'home.html', context)

def chathome(request):
    context = {}
    return render(request, 'chat.html', context)
    

@csrf_exempt
def chatanswer(request):
    context = {}
    chattext = request.GET['ctext']
   
    context['result'] = chattext

    tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2', 
                                              bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                                              pad_token='<pad>', mask_token='<mask>')

    saving_model = TFGPT2LMHeadModel.from_pretrained('static/model.h5')

    sent = '' + chattext + ''
    input_ids = [tokenizer.bos_token_id] + tokenizer.encode(sent) + [tokenizer.pad_token_id]
    input_ids = tf.convert_to_tensor([input_ids])
    output = saving_model.generate(input_ids, max_length=96, do_sample=True, top_k=2)
    sentence = tokenizer.decode(output[0].numpy().tolist())

    context['anstext'] = sentence.split('<pad>')[1][:-5]    
    
    return JsonResponse(context, content_type = 'application/json')


# print(return_answer_by_chatbot('자전거 도로 알려줘'))
