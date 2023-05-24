# -*- coding: UTF-8 -*-
import os
from transformers import AutoTokenizer
import nltk.data
from nltk.stem.snowball import SnowballStemmer
from bart_proc import BartParser
from pubtator_proc import PubTatorParser
from augmenter import AugmentingTokenizer
from labels_prepare import prepare_cros_re_meta_ru
import json

FRAGMENT_TOKENS_CLASS = {'sep':0, 'ent':1, 'usual':2}

class Preprocesser(BartParser, PubTatorParser):
    def __init__(self, tokenizer_, sent_tokenizer_, rel_ids_, max_input=800, count_special=False,
                 count_intersected=True, add_last_ch=False, ents_type_ids=None, use_ents_ids=False,
                 join_same_evids=True):
        PubTatorParser.__init__(self, tokenizer=tokenizer_, max_seq_len=max_input)
        self.tokenizer = tokenizer_
        self.tmp_croped = 0
        self.tmp_not_croped = 0
        self.max_input = max_input
        self.count_special = count_special
        self.count_intersected = count_intersected
        self.sent_tokenizer = sent_tokenizer_
        self.add_last_ch = add_last_ch  # count last * for evids
        self.join_same_evids = join_same_evids

        # --- statistics
        self.rels_in_directory = 0
        self.rels_in_document = 0
        self.rels_found_in_directory = 0
        self.rels_found_in_document = 0
        self.rels_start_count_flag = True

        self.ents = 0
        self.ents_found = 0
        self.rel_ids = rel_ids_
        self.ents_type_ids = ents_type_ids
        self.use_ents_ids=use_ents_ids

        self.not_counted_rels = 0

    @staticmethod
    def add_to_dict(d, e1, e2):
        if e1 in d:
            d[e1].append(e2)
        else:
            d[e1] = [e2]

    @staticmethod
    def remove_from_dict(d, key, val):
        if key in d:
            if len(d[key]) == 1 and d[key][0] == val:
                d.pop(key)
            elif val in d[key]:
                d[key].remove(val)

    @staticmethod
    def count_rels(d):
        counter = 0
        for rel in d.values():
            counter += len(rel)
        return counter

    @staticmethod
    def scroll_token(text, tokens, tokens_iter, digit_iter):
        if text[digit_iter].isspace():
            digit_iter += 1
        elif tokens[tokens_iter].startswith('##'):
            digit_iter += len(tokens[tokens_iter]) - 2
            tokens_iter += 1
        elif tokens[tokens_iter] == '[UNK]':
            while tokens[tokens_iter] == '[UNK]':
                tokens_iter += 1
            next_dig = tokens[tokens_iter][0] if tokens[tokens_iter][0] != '#' else tokens[tokens_iter][2]
            while text[digit_iter] != next_dig:
                digit_iter += 1
        else:
            digit_iter += len(tokens[tokens_iter])
            tokens_iter += 1
        return tokens_iter, digit_iter

    def proc_sent(self, sents_list, cur_sent_id, evids_list, cur_evid_id):
        # todo - count original evids
        # [*, ev, ##id]
        """evids_list: [(name, beg, end, ann_id, evid_type)]"""
        evids_in_sent = []
        running_evids = {}
        token_iter = 0
        cur_sent = sents_list[cur_sent_id]
        cur_sent_tokens = self.tokenizer.tokenize(cur_sent)
        while True:
            if cur_sent_tokens[token_iter] == '@' or cur_sent_tokens[token_iter] == '^':
                digs = ''
                while True:
                    digs += cur_sent_tokens[token_iter+1]
                    cur_sent_tokens.pop(token_iter+1)
                    if cur_sent_tokens[token_iter+1] == '^' or cur_sent_tokens[token_iter+1] == '@':
                        cur_sent_tokens.pop(token_iter + 1)
                        break
                evid_id = int(digs)
                if evid_id in running_evids:
                    evid_beg = running_evids[evid_id]
                    evids_in_sent.append([evids_list[evid_id][0], evid_beg,
                                      token_iter+1, evids_list[evid_id][-2],  evids_list[evid_id][-1]])
                    running_evids.pop(evid_id)
                else:
                    running_evids[evid_id] = token_iter
                token_iter += 1
            else:
                token_iter += 1
                if token_iter >= len(cur_sent_tokens)-1 and len(running_evids) != 0:
                    cur_sent_id += 1
                    cur_sent_tokens += self.tokenizer.tokenize(sents_list[cur_sent_id])
                elif token_iter >= len(cur_sent_tokens)-1:# or cur_evid_id >= len(evids_list):
                    break

        # only for one-token-size aliases
        for evid in evids_in_sent:
            n, beg, end, _, e_type = evid
            if self.use_ents_ids:
                #todo - add multi_token_ids
                e_type_alias = self.ents_type_ids[e_type]
            else:
                e_type_alias = '*'
            cur_sent_tokens[beg], cur_sent_tokens[end-1] = e_type_alias, e_type_alias
        cur_sent_id += 1
        cur_sent_mask = self.build_evid_mask(cur_sent_tokens, evids_in_sent)
        return cur_sent_id, cur_evid_id, evids_in_sent, cur_sent_tokens, cur_sent_mask

    def build_evid_mask(self, tokens_list, evid_in_sent):
        # 0 - sep_token; 1 - usual; 2 - part of entity
        sep_tokens_set = set(self.ents_type_ids) if self.use_ents_ids else set('#')
        mask = []
        for token in tokens_list:
            if token in sep_tokens_set:
                mask.append(FRAGMENT_TOKENS_CLASS['sep'])
            else:
                mask.append(FRAGMENT_TOKENS_CLASS['usual'])
        for evid in evid_in_sent:
            _, beg, end, _, _ = evid
            for i in range(beg+1, end-1):
                mask[i] = FRAGMENT_TOKENS_CLASS['ent']
        return mask



    @staticmethod
    def select_not_intersected_evids(evid_list):
        """[(name, beg, end, ann_id, type)]"""
        evid_list = sorted(evid_list, key=lambda x: (x[1], -x[2]))
        not_intersected = []
        prev_beg, prev_end = -1, -1
        for evid in evid_list:
            cur_beg, cur_end = evid[1], evid[2]
            if cur_beg > prev_end:
                not_intersected.append(evid)
                prev_end = cur_end
            else:
                continue
        return not_intersected

    @staticmethod
    def insert_boarders(text, evid_list):
        boarders_list = []
        for evid_num, evid in enumerate(evid_list[::-1]):
            beg, end = evid[1], evid[2]
            if end == len(text):
                end -= 1
            elif text[end].isalpha():
                end += 1
            boarders_list.append((end, (len(evid_list) - 1) - evid_num, '@'))
            boarders_list.append((beg, (len(evid_list) - 1) - evid_num, '^'))

        boarders_list.sort(key=lambda x: (-x[0], x[1]))
        for (pos, ent_id, c) in boarders_list:
            text = text[:pos] + c + str(ent_id) + c + text[pos:]
        return text

    def split_txt_to_fragments(self, text, evids):
        """splits text in fragments of length < max_input"""

        allowed_evids_list = sorted(evids, key=lambda x: (x[1], x[2])) if \
            self.count_intersected else self.select_not_intersected_evids(evids)

        text = self.insert_boarders(text, allowed_evids_list)
        sents = self.sent_tokenizer.tokenize(text)
        fragment_start, offset_for_tokens_in_fragments, cur_sent_id, cur_evid_id = 0, 0, 0, 0
        converted_evids_in_fragments, tokens_in_fragments, fragment_token_mask = [[]], [[]], [[]]
        cur_fragment_id = 0

        if self.count_special:
            tokens_in_fragments[0] = [self.tokenizer.convert_tokens_to_ids('[CLS]')]
            fragment_token_mask[0] = [FRAGMENT_TOKENS_CLASS['sep']]
            offset_for_tokens_in_fragments += 1

        while cur_sent_id < len(sents):
            cur_sent_id, cur_evid_id, evids_in_sent, tokens, mask = \
                self.proc_sent(sents, cur_sent_id, allowed_evids_list, cur_evid_id)
            tokens = self.tokenizer.convert_tokens_to_ids(tokens)

            if len(tokens_in_fragments[cur_fragment_id]) + len(tokens) >= self.max_input-2:
                cur_fragment_id += 1
                converted_evids_in_fragments.append([])
                if self.count_special:
                    tokens_in_fragments[-1].append(self.tokenizer.convert_tokens_to_ids('[SEP]'))
                    fragment_token_mask[-1].append(FRAGMENT_TOKENS_CLASS['sep'])
                    tokens_in_fragments.append([self.tokenizer.convert_tokens_to_ids('[CLS]')])
                    fragment_token_mask.append([FRAGMENT_TOKENS_CLASS['sep']])
                    offset_for_tokens_in_fragments = 0
                else:
                    tokens_in_fragments.append([])
                    fragment_token_mask.append([])
                    offset_for_tokens_in_fragments = 0

            for evid in evids_in_sent:
                evid[1] += offset_for_tokens_in_fragments
                evid[2] += offset_for_tokens_in_fragments

            tokens_in_fragments[cur_fragment_id] += tokens
            fragment_token_mask[cur_fragment_id] += mask
            converted_evids_in_fragments[-1] += evids_in_sent
            offset_for_tokens_in_fragments += len(tokens)
        if self.count_special:
            tokens_in_fragments[-1].append(self.tokenizer.convert_tokens_to_ids('[SEP]'))
            fragment_token_mask[-1].append(FRAGMENT_TOKENS_CLASS['sep'])
        return tokens_in_fragments, converted_evids_in_fragments, fragment_token_mask

    def convert_evids_to_enst(self, evids, name_aliases, id_aliases):
        ents_dict = {}  # {id:[evids]}
        evid_name_to_ent_id = {}
        ann_id_to_ent_id = {}
        stemmer = SnowballStemmer("russian")
        for evid in evids:
            ent_id = len(ents_dict)
            if self.join_same_evids:
                name, ann_id = stemmer.stem(evid[0].lower()), evid[3]
            else:
                name, ann_id = evid[0].lower(), evid[3]
            if ann_id in ann_id_to_ent_id:
                ent_id = ann_id_to_ent_id[ann_id]
                self.add_to_dict(ents_dict, ent_id, evid)

            elif ann_id in id_aliases:
                for al_ann_id in id_aliases[ann_id]:
                    if al_ann_id in ann_id_to_ent_id:
                        ent_id = ann_id_to_ent_id[al_ann_id]
                        self.add_to_dict(ents_dict, ent_id, evid)
                        break

            elif name in name_aliases:
                for al_name in name_aliases[name]:
                    if al_name in evid_name_to_ent_id:
                        ent_id = evid_name_to_ent_id[al_name]
                        self.add_to_dict(ents_dict, ent_id, evid)
                        break
            elif self.join_same_evids and name in evid_name_to_ent_id:
                ent_id = evid_name_to_ent_id[name]
                self.add_to_dict(ents_dict, ent_id, evid)

            else:
                self.add_to_dict(ents_dict, ent_id, evid)
            ann_id_to_ent_id[ann_id] = ent_id
            evid_name_to_ent_id[name] = ent_id
        return ents_dict

    def generate_ents_combinations(self, ents, rels):
        rels_st = self.count_rels(rels)
        n_ent = len(ents.keys())
        hts = []
        labels = []
        for i in range(n_ent):
            for j in range(n_ent):
                if i == j:
                    continue
                hts.append([i, j])
                mask = [0 for _ in range(len(self.rel_ids) + 1)]
                for evid_i in ents[i]:
                    for evid_j in ents[j]:
                        if (evid_i[3], evid_j[3]) in rels:
                            for rel_name in rels[(evid_i[3], evid_j[3])]:
                                mask[self.rel_ids[rel_name]] = 1
                                self.remove_from_dict(rels, (evid_i[3], evid_j[3]), rel_name)
                if sum(mask) == 0:
                    mask[-1] = 1
                labels.append(mask)

        self.rels_found_in_document += rels_st - self.count_rels(rels)
        if self.count_rels(rels) > 0:
            i = 10
        return hts, labels

    def proc_fragment(self, ann_input, evids, rels_in_doc_set):
        # idxs_names_map = {}  # ent id (T_N) <-> name
        # ents = {}  # name -> number (to convert to idxs)
        rels = {}  # {evid1 : [(evid2, type)]}
        # rel_names = []  # rel_names_list (to convert to idxs)
        name_aliases, id_aliases = {}, {}

        for line in ann_input:
            line = line.strip()

            if line[0] == "R":
                if line in rels_in_doc_set:
                    rels_in_doc_set.remove(line)
                if self.rels_start_count_flag:
                    self.rels_in_document += 1
                self.on_rel_line_bart(line, rels, id_aliases)

            elif line.startswith('*\tAlias'):
                self.on_alias_line_bart(line, name_aliases)

        self.rels_start_count_flag = False
        ents = self.convert_evids_to_enst(evids, name_aliases, id_aliases)

        entity_pos = [[[ev[1], ev[2]] for ev in vals] for vals in ents.values()]
        hts, labels = self.generate_ents_combinations(ents,  rels)
        return entity_pos, hts, labels

    def proc_doc_bart(self, ann_input, txt_input):
        """process input text and annotation to get list of"""
        evids = []
        # count ony entity pos:
        for line in ann_input:
            if line[0] == "T":
                self.on_evid_line_bart(line, evids)

        # process txt file
        tokens_in_fragments, evids_in_fragments, fragments_tokens_mask = self.split_txt_to_fragments(txt_input, evids)

        res = []
        self.rels_start_count_flag = True
        self.rels_in_document = 0
        self.rels_found_in_document = 0
        # count not found rels

        rels_in_doc_set = set([line for line in ann_input if line.startswith("R")])

        for fragment_id in range(len(tokens_in_fragments)):
            tokens = tokens_in_fragments[fragment_id]
            evids = evids_in_fragments[fragment_id]
            mask = fragments_tokens_mask[fragment_id]

            evids_pos, hts, labels = self.proc_fragment(ann_input, evids, rels_in_doc_set)
            if len(hts) > 0:
                res.append({'input_ids': tokens, 'entity_pos': evids_pos, 'hts': hts, 'labels': labels})

        self.not_counted_rels += len(rels_in_doc_set)
        self.rels_in_directory += self.rels_in_document
        self.rels_found_in_directory += self.rels_found_in_document
        return res

    def  proc_input(self, input_type, train_dir, doc_list):
        train_features = None
        if input_type == 'Bart':
            train_features = self.proc_bart_dir(train_dir, doc_list)
        elif input_type == 'pubTator':
            train_features = self.proc_pub_tator_file(train_dir)
        elif input_type == 'pubTatorFilterd':
            train_features = self.proc_pub_tator_filterd_file(train_dir)
        return train_features
