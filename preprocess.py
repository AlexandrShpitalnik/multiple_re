iimport os
from transformers import AutoTokenizer
import nltk.data

# todo - general statistics for entities in files
# todo - add rel_ids
# todo - statistics found rels


class Preprocesser:
    def __init__(self, tokenizer_, sent_tokenizer_, rel_ids_, max_input=1024, count_special=False):
        self.tokenizer = tokenizer_
        self.tmp_croped = 0
        self.tmp_not_croped = 0
        self.max_input = max_input
        self.count_special = count_special
        self.sent_tokenizer = sent_tokenizer_
        # --- statistics
        self.rels_in_directory = 0
        self.rels_in_document = 0
        self.rels_found_in_directory = 0
        self.rels_found_in_document = 0
        self.rels_start_count_flag = True
        self.ents = 0
        self.ents_found = 0
        self.rel_ids = rel_ids_
        #self.rel_ids = {"Body_location_rel": 0, "Severity_rel": 1, "Course_rel": 2, "Modificator_rel": 3,
        #       "Symptom_bdyloc_rel": 4}
        #self.rel_ids = {"CID": 0}

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

    def proc_sent(self, sents_list, cur_sent_id, evids_list, cur_evid_id, offset):
        """evids_list: [(name, beg, end, ann_id)]"""
        evids_in_sent = []
        token_iter = 0
        cur_sent = sents_list[cur_sent_id]
        cur_sent_tokens = self.tokenizer.tokenize(cur_sent)
        evid_started_flag = False
        evid_beg = 0

        while cur_evid_id < len(evids_list) and token_iter < len(cur_sent_tokens)-1:
            if cur_sent_tokens[token_iter] == '^' and not evid_started_flag:
                cur_sent_tokens[token_iter] = '*'
                evid_started_flag = True
                evid_beg = token_iter
                while '^' not in cur_sent_tokens[token_iter+1:]:
                    cur_sent_id += 1
                    cur_sent_tokens += self.tokenizer.tokenize(sents_list[cur_sent_id])
            elif cur_sent_tokens[token_iter] == '^':
                cur_sent_tokens[token_iter] = '*'
                evids_in_sent.append((evids_list[cur_evid_id][0], offset+evid_beg,
                                      offset+token_iter, evids_list[cur_evid_id][-1]))
                cur_evid_id += 1
                evid_started_flag = False
            token_iter += 1

        return cur_sent_id, cur_evid_id, evids_in_sent, cur_sent_tokens

    @staticmethod
    def select_not_intersected_evids(evid_list):
        """[(name, beg, end, ann_id)]"""
        evid_list = sorted(evid_list, key=lambda x: (x[1], -x[2]))
        not_intersected = []
        prev_beg, prev_end = -1, -1
        for evid in evid_list:
            cur_beg, cur_end = evid[1], evid[2]
            if cur_beg > prev_end:
                not_intersected.append(evid)
                prev_end = cur_end
            else:
                # todo - count skipped
                continue
        return not_intersected

    @staticmethod
    def insert_boarders(text, evid_list):
        for evid in evid_list[::-1]:
            beg, end = evid[1], evid[2]
            if end == len(text):
                end -= 1
            elif text[end].isalpha():
                end += 1
            text = text[:end] + '^' + text[end:]
            text = text[:beg] + '^' + text[beg:]
        return text

    def split_txt_to_fragments(self, text, evids):
        """splits text in fragments of length < max_input"""
        allowed_evids_list = self.select_not_intersected_evids(evids)

        text = self.insert_boarders(text, allowed_evids_list)
        sents = self.sent_tokenizer.tokenize(text)
        fragment_start, offset_for_tokens_in_fragments, cur_sent_id, cur_evid_id = 0, 0, 0, 0
        converted_evids_in_fragments, tokens_in_fragments = [[]], [[]]
        cur_fragment_id = 0

        if self.count_special:
            tokens_in_fragments[0] = [self.tokenizer.convert_tokens_to_ids('[CLS]')]
            offset_for_tokens_in_fragments += 1

        while cur_sent_id < len(sents):
            cur_sent_id, cur_evid_id, evids_in_sent, tokens = \
                self.proc_sent(sents, cur_sent_id, allowed_evids_list, cur_evid_id, offset_for_tokens_in_fragments)
            tokens = self.tokenizer.convert_tokens_to_ids(tokens)

            if len(tokens_in_fragments[cur_fragment_id]) + len(tokens) >= self.max_input-2:

                cur_fragment_id += 1
                converted_evids_in_fragments.append([])
                if self.count_special:
                    tokens_in_fragments[-1].append([self.tokenizer.convert_tokens_to_ids('[CLS]')])
                    tokens_in_fragments.append([self.tokenizer.convert_tokens_to_ids('[CLS]')])
                    offset_for_tokens_in_fragments = 1
                else:
                    tokens_in_fragments.append([])
                    offset_for_tokens_in_fragments = 0

            tokens_in_fragments[cur_fragment_id] += tokens
            converted_evids_in_fragments[-1] += evids_in_sent
            offset_for_tokens_in_fragments += len(tokens)
            cur_sent_id += 1
        return tokens_in_fragments, converted_evids_in_fragments

    @staticmethod
    def on_rel_line_bart(line, rels):
        line = line.strip()
        idx, meta = line.split('\t')
        rel_name, evid1, evid2 = meta.split(' ')
        evid1_id, evid2_id = evid1.split(':')[1], evid2.split(':')[1]
        if (evid1_id, evid2_id) in rels:
            rels[(evid1_id, evid2_id)].append(rel_name)
        else:
            rels[(evid1_id, evid2_id)] = [rel_name]

    @staticmethod
    def on_evid_line_bart(line, evid_list):
        line = line.strip()
        line_parts = line.split("\t")
        # ann_idx may be unique or not / same ents may have different names
        ann_idx, loc, name = line_parts[0], line_parts[1], ' '.join(line_parts[2:])
        loc = loc.split(';')[0]
        ent_type, beg, end = loc.split(" ")
        start_num, end_num = min(int(beg), int(end)), max(int(beg), int(end))
        evid_list.append((name, start_num, end_num, ann_idx))

    def on_alias_line_bart(self, line, aliases):
        line = line.strip()
        names = line.split(" ")
        for name in names[2:]:
            self.add_to_dict(aliases, names[1].lower(), name.lower())

    def convert_evids_to_enst(self, evids, aliases):
        ents_dict = {}  # {id:[evids]}
        evid_name_to_ent_id = {}
        ann_id_to_ent_id = {}
        for evid in evids:
            ent_id = len(ents_dict)
            name, ann_id = evid[0].lower(), evid[3]
            if ann_id in ann_id_to_ent_id:
                ent_id = ann_id_to_ent_id[ann_id]
                self.add_to_dict(ents_dict, ent_id, evid)

            elif name in evid_name_to_ent_id:
                ent_id = evid_name_to_ent_id[name]
                self.add_to_dict(ents_dict, ent_id, evid)

            elif name in aliases:
                for al_name in aliases[name]:
                    if al_name in evid_name_to_ent_id:
                        ent_id = evid_name_to_ent_id[al_name]
                        self.add_to_dict(ents_dict, ent_id, evid)
                        evid_name_to_ent_id[name] = ent_id
                        break

            else:
                self.add_to_dict(ents_dict, ent_id, evid)
                ann_id_to_ent_id[ann_id] = ent_id
                evid_name_to_ent_id[name] = ent_id
        return ents_dict

    def generate_ents_combinations(self, ents, rels):
        """ents: {ent_id:[evids]}
        rels: {(evid1, evid2):[names] }"""
        rels_st = self.count_rels(rels)
        if rels_st > 8:
            rels_st += 1
            rels_st -= 1
            pass
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

    def proc_fragment(self, ann_input, evids):
        # idxs_names_map = {}  # ent id (T_N) <-> name
        # ents = {}  # name -> number (to convert to idxs)
        rels = {}  # {evid1 : [(evid2, type)]}
        # rel_names = []  # rel_names_list (to convert to idxs)
        aliases = {}

        for line in ann_input:
            line = line.strip()

            if line[0] == "R":
                if self.rels_start_count_flag:
                    self.rels_in_document += 1
                self.on_rel_line_bart(line, rels)

            elif line.startswith('*\tAlias'):
                self.on_alias_line_bart(line, aliases)

        self.rels_start_count_flag = False
        ents = self.convert_evids_to_enst(evids, aliases)

        entity_pos = [[[ev[1], ev[2]] for ev in vals] for vals in ents.values()]
        hts, labels = self.generate_ents_combinations(ents,  rels)
        return entity_pos, hts, labels

    def pubTator_to_bart(self, file):
        ann_lines = []
        n_rels = 1
        alias_dict = {}
        reading_status = 0
        text = ""
        text_num = ""
        for line in file:
            parts = line.split("|")
            if reading_status < 2:
                assert len(parts) > 2 and (parts[1] == "t" or parts[1] == "a")
                text += "|".join(parts[2:])
                text_num = parts[0]
                reading_status += 1
            elif len(line) > 1:
                rec_parts = line.strip().split("\t")
                assert rec_parts[0] == text_num
                if rec_parts[1].isdigit():
                    ann_lines.append(self.on_ents_line_pub_tator(rec_parts, alias_dict))
                elif rec_parts[1].isalpha():
                    ann_lines.append(self.on_rels_line_pub_tator(rec_parts, alias_dict, n_rels))
                    n_rels += 1
            elif line == "\n":
                return text, ann_lines
        return text, ann_lines

    @staticmethod
    def on_ents_line_pub_tator(line_parts, alias_dict):
        ent_start, ent_end, name, ent_type, ent_id = line_parts[1], line_parts[2], line_parts[3], line_parts[4], \
                                                     line_parts[5]
        ent_id = 'T'+ent_id
        id_parts = ent_id.split('|')
        if len(id_parts) > 1:
            main_id = id_parts[0]
            for alias in id_parts[1:]:
                alias_dict[alias] = main_id
            ent_id = main_id
        pos = " ".join([ent_type, ent_start, ent_end])
        return "\t".join([ent_id, pos, name])

    @staticmethod
    def on_rels_line_pub_tator(line_parts, alias_dict, n_rels):
        rel_type, arg1, arg2 = line_parts[1], line_parts[2], line_parts[3]
        arg1 = "T"+alias_dict[arg1] if arg1 in alias_dict else "T"+arg1
        arg2 = "T"+alias_dict[arg2] if arg2 in alias_dict else "T"+arg2
        assert arg1 != arg2
        return "R"+str(n_rels)+"\t"+rel_type+" "+"Arg1:"+arg1+" Arg2:"+arg2

    def proc_doc_bart(self, ann_input, txt_input):
        """process input text and annotation to get list of"""
        evids = []
        # count ony entity pos:
        for line in ann_input:
            if line[0] == "T":
                self.on_evid_line_bart(line, evids)

        # process txt file
        tokens_in_fragments, evids_in_fragments = self.split_txt_to_fragments(txt_input, evids)

        res = []
        self.rels_start_count_flag = True
        self.rels_in_document = 0
        self.rels_found_in_document = 0
        # todo - ents

        for fragment_id in range(len(tokens_in_fragments)):
            tokens = tokens_in_fragments[fragment_id]
            evids = evids_in_fragments[fragment_id]
            evids_pos, hts, labels = self.proc_fragment(ann_input, evids)
            if len(hts) > 0:
                res.append({'input_ids': tokens, 'entity_pos': evids_pos, 'hts': hts, 'labels': labels})

        # print('r', self.rels_in_document, self.rels_found_in_document, len(tokens_in_fragments))
        self.rels_in_directory += self.rels_in_document
        self.rels_found_in_directory += self.rels_found_in_document
        return res

    def proc_bart_dir(self, w_dir='.'):
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
            text_input, ann_input = txt_file.read(), [line.strip() for line in ann_file]
            res += self.proc_doc_bart(ann_input, text_input)
        print(self.rels_in_directory, self.rels_found_in_directory)
        return res

    def proc_pub_tator_file(self, in_file_name):
        res = []
        with open(in_file_name) as in_file:
            while True:
                text, ann = self.pubTator_to_bart(in_file)
                if text == '':
                    break
                res += self.proc_doc_bart(ann, text)
        print(self.rels_in_directory, self.rels_found_in_directory)
        return res


if __name__ == '__main__':
    nltk.download('punkt')
    rel_ids = {"Body_location_rel": 0, "Severity_rel": 1, "Course_rel": 2, "Modificator_rel": 3,
               "Symptom_bdyloc_rel": 4}
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    sent_tokenizer = nltk.data.load('tokenizers/punkt/russian.pickle')
    prepr = Preprocesser(tokenizer, sent_tokenizer)
    res = prepr.proc_bart_dir("/home/a-shp/Documents/MSU/курсовая/Bert_for_cls/all")
    #res = prepr.proc_pub_tator_file("/home/a-shp/Documents/MSU/курсовая/Bert_for_cls/cdr_dataset/CDR_dev.txt")
