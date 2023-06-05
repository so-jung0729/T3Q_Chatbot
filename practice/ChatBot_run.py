import pandas as pd
import tqdm     # 반복문 진행률

import tensorflow as tf
from transformers import AutoTokenizer
from transformers import TFGPT2LMHeadModel
import torch

tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2', bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>')
model = TFGPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2', from_pt=True)

saving_model = TFGPT2LMHeadModel.from_pretrained('chatbotModel\model.h5')
def return_answer_by_chatbot(user_text):
    sent = '' + user_text + '//'
    input_ids = [tokenizer.bos_token_id] + tokenizer.encode(sent) + [tokenizer.eos_token_id]
    # input_ids = tokenizer.encode(sent)
    input_ids = tf.convert_to_tensor([input_ids])
    output = saving_model.generate(input_ids, max_length=96, do_sample=True, top_k=2)
    sentence = tokenizer.decode(output[0].numpy().tolist())
    chatbot_response = sentence.split('/s')
    return chatbot_response


def answer(q):
  print(return_answer_by_chatbot(q))

print(return_answer_by_chatbot('눈꽃을 볼 수 있는 산을 추천해줘'))