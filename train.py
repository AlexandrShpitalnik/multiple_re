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


def train(args, model, train_features, dev_features, test_features):
    def finetune(features, optimizer, num_epoch, num_steps):
        best_score = -1
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        train_iterator = range(int(num_epoch))
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
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
                if (step + 1) == len(train_dataloader) - 1 or (args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                    dev_score = evaluate(args, model, dev_features, tag="dev")
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
    dev_score = evaluate(args, model, test_features, tag="dev")
    return dev_score


def evaluate(args, model, features, tag="dev"):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    true_labels = []
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
    #preds = np.concatenate(preds, axis=0).astype(np.float32)
    preds = np.concatenate(preds).astype(np.float32)
    #true_labels = np.concatenate(np.array(true_labels), axis=0).astype(np.float32)
    true_labels = np.concatenate(true_labels).astype(np.float32)

    f1 = f1_for_ents(true_labels, preds)

    return f1


def f1_for_ents(true_lbl, pred_lbl):
    true_rels, pred_rels = [], []
    for i in range(len(true_lbl)):
        true_set, pred_set = true_lbl[i], pred_lbl[i]
        for j in range(len(true_set)):
            if j != len(true_set) - 1 and (true_set[j] == 1 or pred_set[j] == 1):
                true_rels.append(true_set[j])
                pred_rels.append(pred_set[j])

    return f1_score(true_rels, pred_rels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_mode", default="", type=str) # cross-validation or standart train/test (cv/st)

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

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    #tokenizer = AutoTokenizer.from_pretrained(
    #    args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    #)
    rel_ids = {"Body_location_rel": 1, "Severity_rel": 2, "Course_rel": 3, "Modificator_rel": 4,
               "Symptom_bdyloc_rel": 5}
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    sent_tokenizer = nltk.data.load('tokenizers/punkt/russian.pickle')

    if args.training_mode == 'cv':
        folds_dirs = os.listdir(path=args.data_dir)
        folds_features = []
        for dir in folds_dirs:
            train_dir = os.path.join(args.data_dir, dir)
            prepr = Preprocesser(tokenizer, sent_tokenizer, rel_ids, max_input=args.max_seq_length)
            train_features = prepr.proc_directory(train_dir)
            folds_features.append(train_features)

        f1_val = []
        for i in range(len(folds_features)):
            validate_set = folds_features[i]
            train_set = folds_features[:i] + folds_features[i+1:]
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
                cur_f1 = train(args, model, train_set, train_set, validate_set)
                f1_val.append(cur_f1)
            else:  # Testing
                model = amp.initialize(model, opt_level="O1", verbosity=0)
                model.load_state_dict(torch.load(args.load_path))
                dev_output = evaluate(args, model, validate_set, tag="dev")
            print(f1_val)
            print(sum(f1_val) / len(f1_val))
            del model
            torch.cuda.empty_cache()


    else:

        train_dir = os.path.join(args.data_dir, args.train_file)
        prepr = Preprocesser(tokenizer, sent_tokenizer, rel_ids, max_input=args.max_seq_length)
        train_features = prepr.proc_directory(train_dir)

        test_dir = os.path.join(args.data_dir, args.test_file)
        prepr = Preprocesser(tokenizer, sent_tokenizer, rel_ids, max_input=args.max_seq_length)
        test_features = prepr.proc_directory(test_dir)

        dev_dir = os.path.join(args.data_dir, args.dev_file)
        prepr = Preprocesser(tokenizer, sent_tokenizer, rel_ids, max_input=args.max_seq_length)
        dev_features = prepr.proc_directory(dev_dir)

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
            train(args, model, train_features, dev_features, test_features)
        else:  # Testing
            model = amp.initialize(model, opt_level="O1", verbosity=0)
            model.load_state_dict(torch.load(args.load_path))
            dev_output = evaluate(args, model, dev_features, tag="dev")

if __name__ == "__main__":
    main()
