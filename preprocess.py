import os
from transformers import AutoTokenizer
import nltk.data
from tqdm import tqdm

#todo - general statistics for entities in files

class Preprocesser:
    def __init__(self, tokenizer, sent_tokenizer, rel_ids, max_input=700, count_special=False):
        self.tokenizer = tokenizer
        self.rel_ids = rel_ids
        self.tmp_croped = 0
        self.tmp_not_croped = 0
        self.max_input = max_input
        self.count_special = count_special
        self.sent_tokenizer = sent_tokenizer
        # --- statistics
        self.rels_in_directory = 0
        self.rels_in_document = 0
        self.rels_found_in_directory = 0
        self.rels_found_in_document = 0
        self.rels_start_count_flag = True
        self.ents = 0
        self.ents_found = 0

    @staticmethod
    def add_to_dict(d, e1, e2):
        if e1 in d:
            d[e1].append(e2)
        else:
            d[e1] = [e2]

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

    def proc_sent(self, sents_list, sent_id, ents_list, ents_iter, offset):
        ents_dict = {}
        ents_idxs = []
        token_iter = 0
        cur_sent = sents_list[sent_id]
        cur_sent_tokens = self.tokenizer.tokenize(cur_sent)
        st_ent_flag = False
        ent_beg = 0

        while ents_iter < len(ents_list) and token_iter < len(cur_sent_tokens)-1:
            word = ents_list[ents_iter][0]
            idx = ents_list[ents_iter][-1]
            if cur_sent_tokens[token_iter] == '^' and not st_ent_flag:
                cur_sent_tokens[token_iter] = '*'
                st_ent_flag = True
                ent_beg = token_iter
                while '^' not in cur_sent_tokens[token_iter+1:]:
                    sent_id += 1
                    cur_sent_tokens += self.tokenizer.tokenize(sents_list[sent_id])
            elif cur_sent_tokens[token_iter] == '^':
                cur_sent_tokens[token_iter] = '*'
                self.add_to_dict(ents_dict, word, (offset+ent_beg, offset+token_iter))
                ents_idxs.append(idx)
                ents_iter += 1
                st_ent_flag = False
            token_iter += 1

        return sent_id, ents_iter, ents_dict, cur_sent_tokens, offset+len(cur_sent_tokens), ents_idxs

    def select_not_intersected_ents(self, ents_list):
        ents_list = sorted(ents_list, key=lambda x: (x[1], -x[2]))
        not_intersected = []
        prev_beg, prev_end = -1, -1
        for ent in ents_list:
            cur_beg, cur_end = ent[1], ent[2]
            if cur_beg > prev_end:
                not_intersected.append(ent)
                prev_end = cur_end
            else:
                # todo - count skipped
                continue
        return not_intersected

    def insert_boarders(self, text, ents_list):
        for ent in ents_list[::-1]:
            beg, end = ent[1], ent[2]
            if end == len(text):
                end -= 1
            elif text[end].isalpha():
                end += 1
            text = text[:end] + '^' + text[end:]
            text = text[:beg] + '^' + text[beg:]
        return text

    def convert_tokens_to_input(self, tokens):
        if self.count_special:
            tokens.append('[SEP]')
            return self.tokenizer.convert_tokens_to_ids(tokens)
        else:
            return self.tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]'])


    def proc_txt(self, in_txt_file, ents_list):
        """splits text in fragments of length < max_input"""
        ents_list = self.select_not_intersected_ents(ents_list)

        text = in_txt_file.read()
        text = self.insert_boarders(text, ents_list)
        sents = self.sent_tokenizer.tokenize(text)
        fragment_start, offset, sent_id, ents_iter = 0, 0, 0, 0
        cur_fragment_id = 0

        ents_fragments = [{}]
        idxs_in_fragments = [[]]
        tokens_fragments = [[]]

        if self.count_special:
            tokens_fragments[cur_fragment_id] = ['[CLS]']
            offset += 1

        while sent_id < len(sents)-1:
            sent_id, ents_iter, ents_dict, tokens, offset, idxs_in_sent = self.proc_sent(sents, sent_id,
                                                                                               ents_list, ents_iter,
                                                                                               offset)
            if len(tokens_fragments[cur_fragment_id]) + len(tokens) < self.max_input-2:
                tokens_fragments[cur_fragment_id] += tokens
                idxs_in_fragments[-1] += idxs_in_sent
                for item in ents_dict.items():
                    if item[0] in ents_fragments[cur_fragment_id]:
                        for it in item[1]:
                            ents_fragments[cur_fragment_id][item[0]].append(it)
                    else:
                        ents_fragments[cur_fragment_id][item[0]] = item[1]
            else:
                idxs_in_fragments.append(idxs_in_sent)
                cur_fragment_id += 1
                tokens_fragments[-1] = self.convert_tokens_to_input(tokens_fragments[-1])
                if self.count_special:
                    tokens_fragments.append(['[CLS]'])
                    offset = 1

                else:
                    tokens_fragments.append([])
                    offset = 0
                ents_fragments.append({})
            sent_id += 1
        if len(tokens_fragments[-1]) > 1:
            tokens_fragments[-1] = self.convert_tokens_to_input(tokens_fragments[-1])
        else:
            pass
            #tokens_fragments.pop(-1)
            # ents_fragments.pop(-1)
        return tokens_fragments, ents_fragments, idxs_in_fragments

    def on_rel_line(self, line, rel_names, rels):
        idx, meta = line.split('\t')
        rel_name, e1, e2 = meta.split(' ')
        if rel_name not in rel_names:
            rel_names.append(rel_name)
        e1_id, e2_id = e1.split(':')[1], e2.split(':')[1]
        if (e1_id, e2_id) in rels:
            if rel_name not in rels[(e1_id, e2_id)]:
                rels[(e1_id, e2_id)].append(rel_name)
        else:
            rels[(e1_id, e2_id)] = [rel_name]

    def on_ents_line(self, line, ents, idxs_names_map, allowed_idxs=None):
        ents_list = []
        line_parts = line.split("\t")
        idx, loc, name = line_parts[0], line_parts[1], ' '.join(line_parts[2:])
        loc = loc.split(';')[0]
        ent_type, beg, end = loc.split(" ")
        start_num, end_num = min(int(beg), int(end)), max(int(beg), int(end))
        if not allowed_idxs:
            ents_list.append([name, start_num, end_num, idx])
            idxs_names_map[idx] = name
            if name not in ents:
                ents[name] = len(ents)
        elif idx in allowed_idxs:
            ents_list.append([name, start_num, end_num])
            idxs_names_map[idx] = name
            if name not in ents:
                ents[name] = len(ents)
        return ents_list

    def on_alias_line(self, line, aliases):
        names = line.split(" ")
        for name in names[2:]:
            self.add_to_dict(aliases, names[1], name)

    def proc_annotation_for_fragment(self, ann_file, ents_locs, idxs):
        idxs_names_map = {}  # ent id (T_N) <-> name
        ents = {}  # name -> number (to convert to idxs)
        rels = {}  # (e1_n, e2_n) -> [r1, r2]
        rel_names = []  # rel_names_list (to convert to idxs)
        aliases = {}

        for line in ann_file:
            line = line.strip()
            if line[0] == "T":
                self.on_ents_line(line, ents, idxs_names_map, idxs)

            elif line[0] == "R":
                if self.rels_start_count_flag:
                    self.rels_in_document += 1
                self.on_rel_line(line, rel_names, rels)

            elif line.startswith('*\tAlias'):
                self.on_alias_line(line, aliases)

        self.rels_start_count_flag = False
        # update names for aliases
        proced_al = []
        for alias_item in aliases.items():
            alias_list = [alias_item[0]] + alias_item[1]
            alias_list = sorted(alias_list)

            # remove not counted entities from aliases
            i = len(alias_list) - 1
            while i >= 0:
                if alias_list[i] not in idxs_names_map:
                    alias_list.pop(i)
                i -= 1

            if len(alias_list) > 0:
                alias_list_names = [idxs_names_map[i] for i in alias_list]
                skip_flag = False
                for a_list in proced_al:
                    if a_list == frozenset(alias_list_names):
                        skip_flag = True
                        break
                if not skip_flag:
                    proced_al.append(frozenset(alias_list_names))
                    main_ent_name = alias_list_names[0]
                    for al in alias_list[1:]:
                        ents_locs[main_ent_name].append(ents_locs.pop(idxs_names_map[al], []))
                        idxs_names_map[al] = main_ent_name

        for tid in idxs_names_map.keys():
            idxs_names_map[tid] = ents[idxs_names_map[tid]]

        # collect entities
        entity_pos = list(ents_locs.values())

        # update names for rels
        ent_relations = {}
        for rel_tt, rel_names in rels.items():
            if rel_tt[0] in idxs_names_map and rel_tt[1] in idxs_names_map:
                rel_a, rel_b = idxs_names_map[rel_tt[0]], idxs_names_map[rel_tt[1]]
                for rel_name in rel_names:
                    rel_name_id = self.rel_ids[rel_name]
                    if (rel_a, rel_b) not in ent_relations:
                        ent_relations[(rel_a, rel_b)] = [rel_name_id]
                    else:
                        ent_relations[(rel_a, rel_b)].append(rel_name_id)
                    self.rels_found_in_document += 1

        # generate all combinations of ents and set labels
        n_ent = len(entity_pos)
        hts = []
        labels = []
        for i in range(n_ent):
            for j in range(n_ent):
                if i == j:
                    continue
                hts.append([i, j])
                mask = [0 for _ in range(len(self.rel_ids) + 1)]
                if (i, j) in ent_relations:
                    for pos in ent_relations[(i, j)]:
                        mask[pos] = 1
                else:
                    mask[-1] = 1
                labels.append(mask)

        return entity_pos, hts, labels

    def proc_doc(self, ann_file, txt_file):
        """process input text and annotation to get list of"""
        ents_list = []
        # count ony entity pos:
        for line in ann_file:
            line = line.strip()
            if line[0] == "T":
                ents_list += self.on_ents_line(line, {}, {})

        # process txt file
        tokens_fragments, ents_fragments, idxs_in_fragments = self.proc_txt(txt_file, ents_list)

        res = []
        self.rels_start_count_flag = True
        self.rels_in_document = 0
        self.rels_found_in_document = 0
        # todo - ents
        for fragment_id in range(len(idxs_in_fragments)):
            if len(idxs_in_fragments[fragment_id]) >= 2 and len(ents_fragments[fragment_id]) >= 2 :
                ents_in_fragment = ents_fragments[fragment_id]
                idxs = idxs_in_fragments[fragment_id]
                tokens_in_fragment = tokens_fragments[fragment_id]
                ann_file.seek(0)
                entity_pos, hts, labels = self.proc_annotation_for_fragment(ann_file, ents_in_fragment, idxs)
                res.append({'input_ids': tokens_in_fragment, 'entity_pos': entity_pos, 'hts': hts, 'labels': labels})
        print('r', self.rels_in_document, self.rels_found_in_document)
        self.rels_in_directory += self.rels_in_document
        self.rels_found_in_directory += self.rels_found_in_document
        return res


    def proc_directory(self, w_dir='.'):
        file_names_list = os.listdir(path=w_dir)
        documents = set()
        res = []
        for file_name in file_names_list:
            doc_id = file_name[:-4]
            documents.add(doc_id)
        for doc in documents:
            print(doc)
            txt, ann = doc + '.txt', doc + '.ann'
            txt_file, ann_file = open(w_dir + '/' + txt), open(w_dir + '/' + ann)
            res += self.proc_doc(ann_file, txt_file)
            txt_file.close()
            ann_file.close()
        print(self.rels_in_directory, self.rels_found_in_directory)
        return res

if __name__ == '__main__':
    rel_ids = {"Body_location_rel":0, "Severity_rel":1, "Course_rel":2, "Modificator_rel":3, "Symptom_bdyloc_rel":4}
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    sent_tokenizer = nltk.data.load('tokenizers/punkt/russian.pickle')
    folds_dirs = os.listdir(path='cv')
    folds_features = []
    for dir in folds_dirs:
        train_dir = os.path.join('cv', dir)
        prepr = Preprocesser(tokenizer, sent_tokenizer, rel_ids, max_input=600)
        train_features = prepr.proc_directory(train_dir)
        folds_features.append(train_features)


def read_data(dir, tokenizer, sent_tokenizer, rel_ids):
    prepr = Preprocesser(tokenizer, sent_tokenizer)
