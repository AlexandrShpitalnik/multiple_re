# -*- coding: UTF-8 -*-
import sys

sys.path.append('/content')
sys.path.append('/content/apex')

import argparse
import os

import numpy as np
import torch
import nltk
from apex import amp
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from model import DocREModel
from utils import set_seed, collate_fn
from preprocess import Preprocesser
from sklearn.metrics import f1_score
import tqdm


class Stat:
    def __init__(self, rel_ids, tokenizer, write_sent=True):
        self.rel_ids = dict(rel_ids)
        self.tokenizer = tokenizer
        self.rel_ids[len(self.rel_ids)] = 'None'
        self.true_examples = []
        self.false_examples = []
        self.write_sent = write_sent

    def count_confusion(self, true_lbls, pred_lbls):
        confusions = [[0, 0, 0, 0] for _ in range(len(self.rel_ids))]  # [[tp, fp, tn, fn]]
        i = 0
        while i < len(true_lbls):
            for rel_type_id in range(len(self.rel_ids)):
                if true_lbls[i][rel_type_id] == pred_lbls[i][rel_type_id] and true_lbls[i][rel_type_id] == 1:
                    confusions[rel_type_id][0] += 1
                elif true_lbls[i][rel_type_id] == pred_lbls[i][rel_type_id] and true_lbls[i][rel_type_id] == 0:
                    confusions[rel_type_id][2] += 1
                elif pred_lbls[i][rel_type_id] == 1:
                    confusions[rel_type_id][1] += 1
                elif pred_lbls[i][rel_type_id] == 0:
                    confusions[rel_type_id][3] += 1
                else:
                    print("!", true_lbls[i], pred_lbls[i])
            i += 1
        return confusions

    def add_predictions_batch(self, inputs, true_labels, ents_pos, hts, pred_labels):
        labels_counter = 0
        for i in range(len(inputs)):
            fragment, t_labels, ents, hts_fragments = inputs[i], true_labels[i], ents_pos[i], hts[i]
            tokens = self.tokenizer.convert_ids_to_tokens(fragment)
            for j in range(len(hts_fragments)):
                e1, e2 = hts_fragments[j]
                sent = self.reconstruct(self.tokenizer, tokens, ents[e1], ents[e2])
                if (t_labels[j] == pred_labels[labels_counter]).all():
                    self.true_examples.append((sent, t_labels[j]))
                    if self.write_sent and t_labels[j][-1] != 1:
                        print(t_labels[j], sent)
                else:
                    self.false_examples.append((sent, t_labels[j], pred_labels[labels_counter]))
                    if self.write_sent:
                        print('pred:', pred_labels[labels_counter],'true:',t_labels[j], sent)
                labels_counter += 1

    @staticmethod
    def reconstruct(tokenizer, tokens, e1, e2):
        tokens = [t for t in tokens if t != '[PAD]']
        if tokens == []:
            print('----')
            return []
        for i, evid in enumerate(e1 + e2):
            beg, end = evid
            try:
                assert tokens[beg] == '*' or tokens[beg] == '^' or tokens[beg] == '@'
                assert tokens[end] == '*' or tokens[end] == '^' or tokens[end] == '@'
                assert end < len(tokens) and beg >= 0
            except:
                print('!!', beg, end, len(tokens), tokens[beg:])

            try:
                if i == 0:
                    tokens[beg], tokens[end] = '^', '^'
                else:
                    tokens[beg], tokens[end] = '@', '@'
            except:
                print('!', beg, len(tokens), tokens[beg:])

        fragment = tokenizer.convert_tokens_to_string(tokens)

        for evid in e1 + e2:
            beg, end = evid
            if beg < len(tokens) and end < len(tokens):
                tokens[beg], tokens[end] = '*', '*'
        return fragment


def train(args, model, train_features, dev_features, test_features, stat, not_found_rels):
    def finetune(features, optimizer, num_epoch, num_steps):
        best_score = -1
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn,
                                      drop_last=True)
        train_iterator = range(int(num_epoch))
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)
        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))
        for epoch in tqdm.tqdm(train_iterator):
            model.zero_grad()
            epoch_loss = []
            for step, batch in enumerate(train_dataloader):
                model.train()
                inputs = {'input_ids': batch[0].to(args.device),
                          'attention_mask': batch[1].to(args.device),
                          'labels': batch[2],
                          'entity_pos': batch[3],
                          'hts': batch[4],
                          }
                outputs = model(**inputs)
                loss = outputs[0] / args.gradient_accumulation_steps
                epoch_loss.append(loss.cpu().detach().numpy())
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if step % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1
                if (step + 1) == len(train_dataloader) - 1 or (
                        args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                    dev_score, _, dev_macro = evaluate(args, model, dev_features, None, not_found_rels, tag="dev")
                    if dev_score > best_score:
                        best_score = dev_score
                        if args.save_path != "":
                            torch.save(model.state_dict(), args.save_path)
            epoch_loss = np.array(epoch_loss)
        return num_steps, model

    new_layer = ["extractor", "bilinear"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": 1e-4},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    num_steps = 0
    set_seed(args)
    model.zero_grad()
    _, model = finetune(train_features, optimizer, args.num_train_epochs, num_steps)
    dev_score, confusions, f1_macro = evaluate(args, model, test_features, stat, not_found_rels, tag="dev")
    return dev_score, confusions, f1_macro


def evaluate(args, model, features, stat, not_found_rels, tag="dev"):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn,
                            drop_last=False)
    preds = []
    true_labels = []
    if stat:
        print(stat.rel_ids)
        print('___________________')
    for batch in dataloader:
        model.eval()
        pred = []

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        with torch.no_grad():

            pred, *_ = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)
            true_labels += batch[2]
            

        if stat:
            stat.add_predictions_batch(batch[0], batch[2], batch[3],
                                       batch[4], pred)
    # preds = np.concatenate(preds, axis=0).astype(np.float32)
    preds = np.concatenate(preds).astype(np.float32)
    # true_labels = np.concatenate(np.array(true_labels), axis=0).astype(np.float32)
    true_labels = np.concatenate(true_labels).astype(np.float32)

    f1 = f1_for_ents(true_labels, preds, not_found_rels)

    confusion = stat.count_confusion(true_labels, preds) if stat else None
    f1_macro = f1_from_confs(confusion) if confusion else 0

    return f1, confusion, f1_macro


def f1_from_confs(confs):
    # [[tp, fp, tn, fn]]
    f1s = []
    for rel_type_stat in confs[:-1]:
        tp, fp, tn, fn = rel_type_stat
        if tp + fp + fn > 0:
            f1s.append((tp) / (tp + 1e-7 + 0.5 * (fp + fn)))
    return sum(f1s) / len(f1s)


def f1_for_ents(true_lbl, pred_lbl, not_found_rels):  # todo - add honest f1 (count wrong last lbl)
    true_rels, pred_rels = [1 for _ in range(not_found_rels)], [0 for _ in range(not_found_rels)]
    for i in range(len(true_lbl)):
        true_set, pred_set = true_lbl[i], pred_lbl[i]
        for j in range(len(true_set)-1):
            #if j != len(true_set) - 1 and (true_set[j] == 1 or pred_set[j] == 1):
            true_rels.append(true_set[j])
            pred_rels.append(pred_set[j])

    assert len(true_lbl) == len(pred_lbl)

    return f1_score(true_rels, pred_rels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_mode", default="", type=str)  # cross-validation or standart train/test (cv/st)
    parser.add_argument("--cv_test_names", default="", type=str)
    parser.add_argument("--input_type", default="", type=str)  # pubTator / bart

    parser.add_argument("--data_dir", default="./dataset/docred", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)

    parser.add_argument("--train_file", default="train_annotated.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    parser.add_argument("--save_path", default="", type=str)
    parser.add_argument("--load_path", default="", type=str)

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=8, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_labels", default=4, type=int,
                        help="Max number of labels in prediction.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=97,
                        help="Number of relation types in dataset.")
                        
    parser.add_argument("--print_sent", type=bool, default=False)

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    # tokenizer = AutoTokenizer.from_pretrained(
    #    args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    # )

    rel_ids = None
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    sent_tokenizer = nltk.data.load('tokenizers/punkt/russian.pickle')

    if args.input_type == 'Bart':
        rel_ids = {"Body_location_rel": 0, "Severity_rel": 1, "Course_rel": 2, "Modificator_rel": 3,
                   "Symptom_bdyloc_rel": 4}
    elif args.input_type == 'pubTator':
        rel_ids = {"CID": 0}

    if args.training_mode == 'cv':
        # BART input
        folds_dirs = os.listdir(path=args.data_dir)
        folds_dirs = sorted([int(i) for i in folds_dirs])
        folds_features = []
        not_found_rels_in_folds = []
        for dir in folds_dirs:
            train_dir = os.path.join(args.data_dir, str(dir))
            prepr = Preprocesser(tokenizer, sent_tokenizer, rel_ids, max_input=args.max_seq_length)
            if args.input_type == 'Bart':
                train_features = prepr.proc_bart_dir(train_dir)
                
                print(train_dir, len(train_features))
                
            elif args.input_type == 'pubTator':
                train_features = prepr.proc_pub_tator_file(train_dir)
            folds_features.append(train_features)
            print(prepr.not_counted_rels)
            not_found_rels_in_folds.append(prepr.not_counted_rels)

        f1_val = []
        f1_macro = []
        test_dirs = args.cv_test_names.split(" ")
        test_dirs = [int(i) for i in test_dirs]
        for i in test_dirs:
            stat = Stat(rel_ids, tokenizer)
            validate_set = folds_features[i]
            train_set = folds_features[:i] + folds_features[i + 1:]
            not_found_rels = not_found_rels_in_folds[i]
            res = []
            for tmp in train_set:
                res += tmp
            train_set = res

            model = AutoModel.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
            )

            config.cls_token_id = tokenizer.cls_token_id
            config.sep_token_id = tokenizer.sep_token_id
            config.transformer_type = args.transformer_type

            set_seed(args)
            model = DocREModel(config, model, num_labels=args.num_labels)
            model.to(0)

            if args.load_path == "":  # Training
                cur_f1, confusions, macro_f1 = train(args, model, train_set, train_set, validate_set, stat, not_found_rels)
                f1_val.append(cur_f1)
                f1_macro.append(macro_f1)
            else:  # Testing
                model = amp.initialize(model, opt_level="O1", verbosity=0)
                model.load_state_dict(torch.load(args.load_path))
                dev_output, confusions, dev_macro = evaluate(args, model, validate_set, stat, not_found_rels, tag="dev")
                print("dir_num:", i, "f1:", cur_f1)
            del model
            torch.cuda.empty_cache()

            print('f1: ', sum(f1_val) / len(f1_val), f1_val)
            print('macro_f1: ', sum(f1_macro) / len(f1_macro), f1_macro)
            print(confusions)

    else:
        # PubTatpor input
        train_dir = os.path.join(args.data_dir, args.train_file)
        prepr = Preprocesser(tokenizer, sent_tokenizer, rel_ids, max_input=args.max_seq_length)
        if args.input_type == 'Bart':
            train_features = prepr.proc_bart_dir(train_dir)
        elif args.input_type == 'pubTator':
            train_features = prepr.proc_pub_tator_file(train_dir)

        test_dir = os.path.join(args.data_dir, args.test_file)
        prepr = Preprocesser(tokenizer, sent_tokenizer, rel_ids, max_input=args.max_seq_length)
        if args.input_type == 'Bart':
            test_features = prepr.proc_bart_dir(test_dir)
        elif args.input_type == 'pubTator':
            test_features = prepr.proc_pub_tator_file(test_dir)
        not_found_rels = prepr.not_counted_rels
        print(not_found_rels)

        dev_dir = os.path.join(args.data_dir, args.dev_file)
        prepr = Preprocesser(tokenizer, sent_tokenizer, rel_ids, max_input=args.max_seq_length)
        if args.input_type == 'Bart':
            dev_features = prepr.proc_bart_dir(dev_dir)
        elif args.input_type == 'pubTator':
            dev_features = prepr.proc_pub_tator_file(dev_dir)

        model = AutoModel.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )

        config.cls_token_id = tokenizer.cls_token_id
        config.sep_token_id = tokenizer.sep_token_id
        config.transformer_type = args.transformer_type

        set_seed(args)
        model = DocREModel(config, model, num_labels=args.num_labels)
        model.to(0)

        f1_val = []
        f1_macro = []
        if args.load_path == "":  # Training
            stat = Stat(rel_ids, tokenizer)
            cur_f1, confusions, macro_f1 = train(args, model, train_features, dev_features, test_features, stat, not_found_rels)
            f1_macro.append(macro_f1)
            f1_val.append(cur_f1)
            print('f1: ', sum(f1_val) / len(f1_val), f1_val)
            print('macro_f1: ', sum(f1_macro) / len(f1_macro), f1_macro)
            print(confusions)

        else:  # Testing
            stat = Stat(rel_ids, tokenizer)
            model = amp.initialize(model, opt_level="O1", verbosity=0)
            model.load_state_dict(torch.load(args.load_path))
            dev_output, _, dev_macro = evaluate(args, model, dev_features, stat, not_found_rels, tag="dev")


if __name__ == "__main__":
    main()
