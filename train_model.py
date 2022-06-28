import gc
import os
import torch
import pickle
import time
import random
import argparse
from pandas import DataFrame
import numpy as np
from random import shuffle
from collections import defaultdict
from tqdm.auto import tqdm
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.nn import functional as F
from transformers import XLMRobertaTokenizer, XLMRobertaForMaskedLM


def encode(tokenizer, sent):
    return tokenizer([sent], return_tensors='pt', padding=True, truncation=True)['input_ids'][0]

class BatchGenerator:
    def __init__(self, prepared_data_by_num_target_tokens, max_batch_len=2000, max_batches_per_epoch=0, shuffle=True):
        for k in prepared_data_by_num_target_tokens:
            prepared_data_by_num_target_tokens[k] = sorted(prepared_data_by_num_target_tokens[k], key=lambda x: len(x[0]))
        batches = []
        for lens, data in prepared_data_by_num_target_tokens.items():
            y_len = lens[0]
            rng = torch.LongTensor(range(y_len))
            cur_batch = []
            cur_batch_len = 0
            for x, y, first_mask_tok_id in data:
                len_x = len(x)
                if lens[1] > max_batch_len:
                    continue
                if cur_batch_len + len_x < max_batch_len:
                    cur_batch_len = cur_batch_len + len_x
                else:
                    cur_batch_len = len_x
                    if cur_batch:
                        batches.append(cur_batch)
                    cur_batch = []
                cur_batch.append((x, y, rng + first_mask_tok_id)) 
            if cur_batch:
                batches.append(cur_batch)
        self.num_batches = len(batches)
        self.X = [torch.tensor([list(s[0]) for s in  batch]) for batch in batches]
        self.y = [torch.tensor([s[1] for s in batch]) for batch in batches]
        self.ids = [torch.tensor([s[2] for s in  batch]) for batch in batches]
        self.valid_part = 0
        self.max_batches_per_epoch = max_batches_per_epoch
        self.shuffle = shuffle

    def get_batch(self):
        batch_id = random.randint(0, int(self.num_batches) - 1)
        return self.X[batch_id], self.y[batch_id], self.ids[batch_id]

    def get_batch_generator(self):
        idxes = list(range(len(self.ids)))
        if self.shuffle:
            random.shuffle(idxes)
        counter = 0
        for idx in idxes:
            yield self.X[idx], self.y[idx], self.ids[idx]
            counter += 1
            if (self.max_batches_per_epoch > 0 and counter > self.max_batches_per_epoch):
                break


def plot(train, valid, lr, metrics, outs):
    for r in outs:
        print(r)
    print("Train loss", train)
    print("Valid loss", valid)
    print("lr", lr)

    
def get_params(model, num_layers=2):
    for i in range(24 - num_layers, 24):
        yield model.roberta.encoder.layer[i].parameters()
    
        
def train_model(model, tokenizer, gen, gen_valid, num_epoch, num_batches_per_epoch, optimizer, scheduler, run_dir):
    train_loss = []
    valid_loss = []
    iters_train = []
    iters_valid = []
    decoded_train = []
    step = 0
    start_time = time.time()
    metrics = defaultdict(list)
    lr = []
    s_loss = 0
    best_loss = -100000
    last_10 = []
    outs = []
    model.train()
    gc.collect()
    for i in range(num_epoch):
        s_loss = 0.0
        counter = 0
        torch.cuda.empty_cache()
        #progress_bar = range(num_batches_per_epoch)
        for X, y, target_ids in list(gen.get_batch_generator()):
            torch.cuda.empty_cache()
            model.zero_grad()
            X = X.to('cuda')
            outputs = model(X)[0]
            preds = outputs[list(range(len(y))), target_ids, :].reshape(len(y), -1)
            if counter % 1000 == 0:
                decoded_train.append((tokenizer.decode([i for i in X[0]]), tokenizer.decode([y[0]]),
                                      tokenizer.decode([preds[0].to('cpu').argmax()])))

            y = y.to('cuda')
            try:
                loss = F.cross_entropy(preds, y)
                loss.backward()
            except Exception as ex:
                print(str(ex), X.shape)
                del loss
                del X
                del preds
                optimizer.zero_grad()
                model.zero_grad()
                t = torch.cuda.get_device_properties(0).total_memory
                r = torch.cuda.memory_reserved(0)
                a = torch.cuda.memory_allocated(0)
                f = r-a  # free inside reserved
                print(f)
                outputs = outputs.to('cpu')
                y = y.to('cpu')
                gc.collect()
                preds = outputs[list(range(len(y))), target_ids, :].reshape(len(y), -1)
                del outputs
                gc.collect()
                for _ in range(5):
                    try:
                        loss = F.cross_entropy(preds, y)
                        loss.backward()
                        break
                    except Exception as ex:
                        print(str(ex), X.shape)
                        t = torch.cuda.get_device_properties(0).total_memory
                        r = torch.cuda.memory_reserved(0)
                        a = torch.cuda.memory_allocated(0)
                        f = r-a
                        print(f)
                        gc.collect()
                        time.sleep(1)
                        gc.collect()               
            optimizer.step()
            optimizer.zero_grad()
            model.zero_grad()
            scheduler.step()
            
            #progress_bar.update(1)
            s_loss += float(loss.item()) * len(y)
            counter += len(y)
            train_loss.append(float(loss))
            iters_train.append(step)
            lr.append(optimizer.state_dict()["param_groups"][0]["lr"])
            step += 1
            if 0 and step % 200 == 0:
                plot(train_loss, valid_loss, lr, metrics, outs[:20] + decoded_train[-10:])
                if i > 0:
                    print(i, time.time() - start_time, train_loss[-1], valid_loss[-1])
            torch.cuda.empty_cache()

        #progress_bar = tqdm(range(len(gen_valid.y)))
        with torch.no_grad():
            model.eval()
            loss = 0
            num_s = 0
            for X, y, target_ids in gen_valid.get_batch_generator():        
                num_s += len(y)
                outputs = model(X.to('cuda'))[0][list(range(len(y))), target_ids, :]
                yc = y.to('cuda')
                loss += float(F.cross_entropy(outputs.reshape(len(y), -1), yc).item()) * len(y)
                #progress_bar.update(1)
            loss /= num_s
            if len(valid_loss) == 0 or loss <= min(valid_loss):
                torch.save(model, run_dir + "/" + 'model_best.pth')
            valid_loss.append(float(loss))
            iters_valid.append(step)
            torch.save(model, run_dir + "/" + 'model_last.pth')
            print("Model saved to ", run_dir + "/" + 'model_last.pth')
            model.train() 
        #shuffle(outs)
        print(i, time.time() - start_time, train_loss[-1], valid_loss[-1], decoded_train[-10:])
        plot(train_loss, valid_loss, lr, metrics, outs[:20] + decoded_train[-10:])
        torch.cuda.empty_cache()
        del X
        del yc
        del outputs
        del loss
        gc.collect()
        gc.collect()
    print(train_loss, valid_loss, lr, metrics, decoded_train[-10:])   
    return 0
                
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", dest="model_name", type=str, default="xlm-roberta-large")
    parser.add_argument("--filenames", dest="filenames", type=str, default="en_ru_0.1_{target}-(Russian:-{masks})_1000")
    parser.add_argument("--layers", dest="layers", type=str, default="all")
    parser.add_argument("--max_lr", dest="max_lr", type=float, default=1e-5)
    parser.add_argument("--max_batch_len", dest="max_batch_len", type=int, default=2000)
    args = parser.parse_args()
    model_name = args.model_name
    max_batch_len = args.max_batch_len
    device = 0
    max_lr = args.max_lr
    num_warmup_steps = 5000
    num_epoch = 20
    num_batches_per_epoch = 5000
    layers = args.layers
    train_filename = "train_" + args.filenames.replace("-", " ")
    valid_filename = "valid_"+ args.filenames.replace("-", " ")
    
    num_training_steps = num_batches_per_epoch * num_epoch
    
    run_dir = train_filename + "_" + model_name + "_" + str(layers) + "_" + str(max_lr)
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)
        
    with open(train_filename, "rb") as f:
        train = pickle.load(f)
    with open(valid_filename, "rb") as f:
        valid = pickle.load(f)

    gen_train = BatchGenerator(train,  max_batches_per_epoch=num_batches_per_epoch, max_batch_len=max_batch_len)
    gen_valid = BatchGenerator(valid, max_batches_per_epoch=num_batches_per_epoch, max_batch_len=max_batch_len)
    
    model = XLMRobertaForMaskedLM.from_pretrained(model_name)
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
    model = model.to('cuda')
    if layers != "all":
        for param in model.parameters():
            param.requires_grad = False
        for layer in get_params(model, int(layers)):
            for param in layer:
                param.requires_grad = True

    optimizer = AdamW(model.parameters(), lr=max_lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    train_model(model, tokenizer, gen_train, gen_valid, num_epoch, num_batches_per_epoch, optimizer, scheduler, run_dir)
