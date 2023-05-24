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
from src.transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from model import DocREModel
from model_balanceloss import DocREModelUnet
from utils import set_seed, collate_fn, build_mask
from preprocess import Preprocesser
from sklearn.metrics import f1_score
from augmenter import AugmentingTokenizer
import tqdm
import random
from labels_prepare import prepare_cros_re_meta_ru, prepare_cros_re_meta_en

rel_ids_set = {
    "CORPUS": {"Body_location_rel": 0, "Severity_rel": 1, "Course_rel": 2, "Modificator_rel": 3,
               "Symptom_bdyloc_rel": 4},
    "CDR": {"CID": 0}
}

ents_ids_set = {
    "CORPUS": {"Disease": "0", "Body_location": "1", "Symptom": "2", 
               "Treatment": "3", "Severity": "4" , "Course": "5", "Drug": "6"},
    "CDR": {"Disease": "0", "Chemical": "1"}
}

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
                        # print(t_labels[j], sent)
                        pass
                else:
                    self.false_examples.append((sent, t_labels[j], pred_labels[labels_counter]))
                    if self.write_sent:
                        pass
                labels_counter += 1

    @staticmethod
    def reconstruct(tokenizer, tokens, e1, e2):
        tokens = [t for t in tokens if t != '[PAD]']
        if tokens == []:
            print('----')
            return []
        for i, evid in enumerate(e1 + e2):
            beg, end = evid
            end -= 1
            try:
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
            end -= 1
            if beg < len(tokens) and end < len(tokens):
                tokens[beg], tokens[end] = '*', '*'
        return fragment


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
        for j in range(len(true_set) - 1):
            true_rels.append(true_set[j])
            pred_rels.append(pred_set[j])

    assert len(true_lbl) == len(pred_lbl)
    return f1_score(true_rels, pred_rels)


def evaluate(args, model, features, stat, not_found_rels, tag="dev"):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn,
                            drop_last=False)
    preds, true_labels = [], []
    for batch in dataloader:
        model.eval()
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
    preds = np.concatenate(preds, axis=0).astype(np.float32)
    true_labels = np.concatenate(true_labels).astype(np.float32)

    f1 = f1_for_ents(true_labels, preds, not_found_rels)

    confusion = stat.count_confusion(true_labels, preds) if stat else None
    f1_macro = f1_from_confs(confusion) if confusion else 0

    return f1, confusion, f1_macro


def train(args, model, train_features, dev_features, test_features, stat, not_found_rels):
    new_layer = ["extractor", "bilinear"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": 1e-4},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    num_backward_steps = 0
    set_seed(args)
    model.zero_grad()

    num_epoch = args.num_train_epochs
    best_dev_score, best_score, best_macro, best_cnf = -1, -1, -1, None
    train_dataloader = DataLoader(train_features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn,
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
            mask = [build_mask(batch[0][i], batch[3][i]) for i in range(len(batch[0]))]
            mask = torch.Tensor(mask)
            inputs = {'input_ids': batch[0].to(args.device),
                      'attention_mask': batch[1].to(args.device),
                      'labels': batch[2],
                      'entity_pos': batch[3],
                      'hts': batch[4],
                      'noise_flag': True,
                      'noise_add_fraq': args.noise_add_fraq,
                      'tokens_mask': mask,
                      'add_noise_type': args.add_noise_type
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
                num_backward_steps += 1
            if (step + 1) == len(train_dataloader) - 1 or (
                    args.evaluation_steps > 0 and num_backward_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                if args.print_best_stat:
                    dev_score, _, _ = evaluate(args, model, dev_features, stat, not_found_rels, tag="dev")
                    if dev_score > best_dev_score:
                        best_dev_score = dev_score
                        best_score, best_cnf, best_maco = evaluate(args, model, test_features, stat, not_found_rels,
                                                                   tag="dev")
                        if args.save_path != "":
                            torch.save(model.state_dict(), args.save_path)
        epoch_loss = np.array(epoch_loss)
    dev_score, confusions, f1_macro = evaluate(args, model, test_features, stat, not_found_rels, tag="dev")
    if args.print_best_stat:
        return best_score, best_cnf, best_macro
    else:
        return dev_score, confusions, f1_macro


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_mode", default="", type=str)  # cross-validation or standart train/test ("cv/st")
    parser.add_argument("--cv_test_names", default="", type=str)  # sub dir test for one fold cv
    parser.add_argument("--cv_dirs_mode", default="", type=str)  # one_fold_cv on multi_fold_cv
    parser.add_argument("--cv_n_folds", default=5, type=int)
    parser.add_argument("--input_type", default="", type=str)  # pubTator / Bart
    parser.add_argument("--print_best_stat", default=False, type=bool)
    parser.add_argument("--join_same_evids", action="store_true")
    parser.add_argument("--pre_train", action="store_true")

    parser.add_argument("--data_dir", default="./dataset/docred", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)
    parser.add_argument("--train_from_saved_model", default="", type=str)

    parser.add_argument("--augment_ratio", default=0.2, type=float)
    parser.add_argument("--add_aug_samples", default=False, type=bool)  # todo - use
    parser.add_argument("--unique_category_sep", default=False, type=bool)
    parser.add_argument("--noise_type", type=str, default="")
    parser.add_argument("--noise_eps", type=float, default=0.0)
    parser.add_argument("--print_sent", type=bool, default=False)
    parser.add_argument("--noise_add_fraq", type=float, default=0.0)
    parser.add_argument("--add_noise_type", type=str, default="all")  # ["all", "not_sep", "usual"]

    parser.add_argument("--train_file", default="train_annotated.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    parser.add_argument("--corpus_labels_type", default="CDR", type=str)
    parser.add_argument("--save_path", default="", type=str)
    parser.add_argument("--load_path", default="", type=str)

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--unet_cls", default="", type=str)
    parser.add_argument("--unet_in_dim", type=int, default=3,
                        help="unet_in_dim.")
    parser.add_argument("--unet_out_dim", type=int, default=256,
                        help="unet_out_dim.")
    parser.add_argument("--down_dim", type=int, default=256,
                        help="down_dim.")
    parser.add_argument("--channel_type", type=str, default='',
                        help="unet_out_dim.")
    parser.add_argument("--log_dir", type=str, default='',
                        help="log.")
    parser.add_argument("--max_height", type=int, default=42,
                        help="log.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=8, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
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
    args = parser.parse_args()
    return args

def get_labels(args, tokenizer):
    ents_ids_set['RU_MED'], rel_ids_set['RU_MED'] = prepare_cros_re_meta_ru(tokenizer)
    ents_ids_set['EN_MED'], rel_ids_set['EN_MED'] = prepare_cros_re_meta_en(tokenizer)
    ds_labels = args.corpus_labels_type
    rel_ids, ents_ids = rel_ids_set[ds_labels], ents_ids_set[ds_labels]
    return rel_ids, ents_ids


def load_model(args, config, tokenizer, num_labels):
    if args.noise_type:
        print("__train_config")
        config.noised_data_config = {"type": args.noise_type, "eps": args.noise_eps}
    else:
        config.noised_data_config = None
    config.output_attentions = True

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config
    )

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type

    set_seed(args)
    if args.unet_cls == "True":
        model = DocREModelUnet(config, args, model, num_labels=1)
    else:
        model = DocREModel(config, model, num_labels=1)
    model.to(0)

    if args.train_from_saved_model != '':
        model.load_state_dict(torch.load(args.train_from_saved_model)["checkpoint"])
        print("load saved model from {}.".format(args.train_from_saved_model))

    return model


def prepare_dataset(cur_dir, args, tokenizer, sent_tokenizer, rel_ids, ents_ids):
    """
    add options for new input formates / rel ids sets
    """
    if type(cur_dir) == list:
        target_dir = args.data_dir
        doc_list = cur_dir
    elif args.input_type == "Bart":
        target_dir = os.path.join(args.data_dir, cur_dir)
        doc_list = None
    else:
        # from pubTator file
        target_dir = os.path.join(args.data_dir, cur_dir)
        doc_list = None
    use_ents_flag = args.unique_category_sep
    prepr = Preprocesser(tokenizer, sent_tokenizer, rel_ids, max_input=args.max_seq_length, ents_type_ids=ents_ids,
                         use_ents_ids=use_ents_flag, count_special=True, join_same_evids=args.join_same_evids)
    features = prepr.proc_input(args.input_type, target_dir, doc_list)
    return features, prepr.not_counted_rels


def train_eval_iteration(args, model, train_set, validate_set, test_set, stat, not_found_rels):
    if args.load_path == "":  # Training
        cur_f1, confusions, macro_f1 = train(args, model, train_set, validate_set, test_set, stat,
                                             not_found_rels)
    else:  # Testing
        model = amp.initialize(model, opt_level="O1", verbosity=0)
        model.load_state_dict(torch.load(args.load_path))
        cur_f1, confusions, macro_f1 = evaluate(args, model, test_set, stat, not_found_rels,
                                                tag="dev")
    del model
    torch.cuda.empty_cache()
    return cur_f1, macro_f1, confusions

def construct_cv_folds(objs, n_folds, is_shuffle=False):
    if is_shuffle:
        random.shuffle(objs)
    fold_size = len(objs) // n_folds
    folds = []
    for i in range(n_folds):
        folds.append(objs[i * fold_size: (i + 1) * fold_size])
    if fold_size * n_folds != len(objs):
        folds[-1].extend(objs[fold_size * n_folds:])
    return folds

def one_dir_cv(args, config, tokenizer, aug_tokenizer, sent_tokenizer, rel_ids, ents_ids):
    # only bart dir
    file_names_list = os.listdir(path=args.data_dir)
    docs = set()
    for file_name in file_names_list:
        doc_id = file_name[:-4]
        docs.add(doc_id)
    docs = list(docs)
    if args.pre_train:
        folds = [docs]
    else:
        folds = construct_cv_folds(docs, args.cv_n_folds, True)

    f1_val, f1_macro, confusions_list = cv_mode(args, config, folds, tokenizer, aug_tokenizer, sent_tokenizer,
                                                rel_ids, ents_ids)
    return f1_val, f1_macro, confusions_list

def several_dirs_cv(args, config, tokenizer, aug_tokenizer, sent_tokenizer, rel_ids, ents_ids):
    folds_dirs = os.listdir(path=args.data_dir)
    folds_dirs = sorted([int(i) for i in folds_dirs])
    f1_val, f1_macro, confusions_list = cv_mode(args, config, folds_dirs, tokenizer, aug_tokenizer, sent_tokenizer,
                                                rel_ids, ents_ids)
    return f1_val, f1_macro, confusions_list

def cv_mode(args, config, folds, tokenizer, aug_tokenizer, sent_tokenizer, rel_ids, ents_ids):
    folds_features_train, not_found_rels_in_folds_train = [], []
    folds_features_test, not_found_rels_in_folds_test = [], []

    for cur_dir in folds:
        cur_dir = str(cur_dir) if type(cur_dir) == int else cur_dir
        cur_features_train, cur_not_proced_train = prepare_dataset(cur_dir, args, aug_tokenizer, sent_tokenizer,
                                                                   rel_ids, ents_ids)
        folds_features_train.append(cur_features_train)
        not_found_rels_in_folds_train.append(cur_not_proced_train)

        cur_features_test, cur_not_proced_test = prepare_dataset(cur_dir, args, tokenizer, sent_tokenizer, rel_ids,
                                                                 ents_ids)
        folds_features_test.append(cur_features_test)
        not_found_rels_in_folds_test.append(cur_not_proced_test)

        if args.add_aug_samples:
            for i in range(len(folds)):
                folds_features_train[i] += folds_features_test[i]

    f1_val, f1_macro, confusions_list = [], [], []
    test_dirs = [int(i) for i in args.cv_test_names.split(" ")]

    if args.pre_train:
        stat = Stat(rel_ids, tokenizer)
        validate_set, train_set = folds_features_test[0], folds_features_test[0]
        model = load_model(args, config, tokenizer, len(rel_ids) + 1)
        not_found_rels = not_found_rels_in_folds_test[0]

        cur_f1, macro_f1, confusions = train_eval_iteration(args, model, train_set, train_set, validate_set, stat,
                                                            not_found_rels)
        print('f1: ', cur_f1)
        print('macro_f1: ', f1_macro)
        print(confusions)

        if args.save_path != "":
            torch.save({
                'epoch': args.num_train_epochs,
                'checkpoint': model.state_dict(),
                'best_f1': cur_f1
                #'optimizer': optimizer.state_dict()
            }, args.save_path
                , _use_new_zipfile_serialization=False)

        return cur_f1, macro_f1, [confusions]

    for i in test_dirs:
        stat = Stat(rel_ids, tokenizer)
        validate_set, train_set = folds_features_test[i], folds_features_train[:i] + folds_features_train[i + 1:]
        not_found_rels = not_found_rels_in_folds_test[i]
        train_set = [item for sublist in train_set for item in sublist]

        model = load_model(args, config, tokenizer, len(rel_ids) + 1)

        cur_f1, macro_f1, confusions = train_eval_iteration(args, model, train_set, train_set, validate_set, stat,
                                                            not_found_rels)
        f1_val.append(cur_f1)
        f1_macro.append(macro_f1)
        confusions_list.append(confusions)

        print('f1: ', sum(f1_val) / len(f1_val), f1_val)
        print('macro_f1: ', sum(f1_macro) / len(f1_macro), f1_macro)
        print(confusions)
    return f1_val, f1_macro, confusions_list

def train_eval_mode(args, config, tokenizer, aug_tokenizer, sent_tokenizer, rel_ids, ents_ids):
    train_features_aug, not_found_rels_aug = prepare_dataset(args.train_file, args, aug_tokenizer, sent_tokenizer,
                                                             rel_ids, ents_ids)
    train_features, not_found_rels = prepare_dataset(args.train_file, args, tokenizer, sent_tokenizer,
                                                     rel_ids, ents_ids)
    print(not_found_rels)
    test_features, not_found_rels_test = prepare_dataset(args.test_file, args, tokenizer, sent_tokenizer, rel_ids,
                                                         ents_ids)
    print(not_found_rels_test)
    dev_features, not_found_rels_dev = prepare_dataset(args.dev_file, args, tokenizer, sent_tokenizer, rel_ids,
                                                       ents_ids)
    dev_features_aug, not_found_rels_dev_aug = prepare_dataset(args.dev_file, args, aug_tokenizer, sent_tokenizer,
                                                               rel_ids, ents_ids)
    print(not_found_rels_dev)

    if args.add_aug_samples:
        test_features += train_features_aug
        dev_features += dev_features_aug

    model = load_model(args, config, tokenizer, len(rel_ids) + 1)

    f1_val = []
    f1_macro = []
    if args.load_path == "":  # Training
        stat = Stat(rel_ids, tokenizer)
        cur_f1, confusions, macro_f1 = train(args, model, train_features, dev_features, test_features, stat,
                                             not_found_rels_test)

        f1_macro.append(macro_f1)
        f1_val.append(cur_f1)
        print('f1: ', sum(f1_val) / len(f1_val), f1_val)
        print('macro_f1: ', sum(f1_macro) / len(f1_macro), f1_macro)
        print(confusions)
        return f1_val, f1_macro, confusions

    else:  # Testing
        stat = Stat(rel_ids, tokenizer)
        model = amp.initialize(model, opt_level="O1", verbosity=0)
        model.load_state_dict(torch.load(args.load_path))
        dev_output, _, dev_macro = evaluate(args, model, dev_features, stat, not_found_rels_dev, tag="dev")
        test_output, _, test_macro = evaluate(args, model, test_features, stat, not_found_rels_test, tag="dev")

        return dev_macro, test_macro

def main():
    args = parse_arguments()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    rel_ids, ents_ids = get_labels(args, tokenizer)
    aug_tokenizer = AugmentingTokenizer(args.model_name_or_path)

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=len(rel_ids) + 1
    )
    sent_tokenizer = nltk.data.load('tokenizers/punkt/russian.pickle')

    if args.training_mode == 'cv':
        if args.cv_dirs_mode == 'one_fold_cv':
            one_dir_cv(args, config, tokenizer, aug_tokenizer, sent_tokenizer, rel_ids, ents_ids)
        else:
            several_dirs_cv(args, config, tokenizer, aug_tokenizer, sent_tokenizer, rel_ids, ents_ids)
    else:
        train_eval_mode(args, config, tokenizer, aug_tokenizer, sent_tokenizer, rel_ids, ents_ids)

if __name__ == "__main__":
    main()
