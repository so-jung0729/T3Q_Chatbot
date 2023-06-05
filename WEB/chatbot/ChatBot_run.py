import pandas as pd
import tqdm     # 반복문 진행률

import tensorflow as tf
from transformers import AutoTokenizer
from transformers import TFGPT2LMHeadModel
import torch


def return_answer_by_chatbot(user_text):
    tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2', bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>')
    model = TFGPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2', from_pt=True)

    saving_model = TFGPT2LMHeadModel.from_pretrained('chatbotModel\model.h5')

    sent = '' + user_text + ''
    input_ids = [tokenizer.bos_token_id] + tokenizer.encode(sent) + [tokenizer.pad_token_id]
    input_ids = tf.convert_to_tensor([input_ids])
    output = saving_model.generate(input_ids, max_length=96, do_sample=True, top_k=2)
    sentence = tokenizer.decode(output[0].numpy().tolist())
    chatbot_response = sentence.split('<pad>')[1][:-5]
    return chatbot_response


print(return_answer_by_chatbot('오늘 휴관일이야?'))