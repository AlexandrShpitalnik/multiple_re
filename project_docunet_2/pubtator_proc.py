
class PubTatorParser:
    def __init__(self, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.cdr_rel2id = {'1:NR:2': 1, '1:CID:2': 0}

    @staticmethod
    def on_ents_line_pub_tator(line_parts):
        ent_start, ent_end = str(min(int(line_parts[1]), int(line_parts[2]))), \
                             str(max(int(line_parts[1]), int(line_parts[2])))
        name, ent_type, ent_id = line_parts[3], line_parts[4], line_parts[5]

        id_parts = ent_id.split('|')
        pos = " ".join([ent_type, ent_start, ent_end])
        res = []
        for i in range(len(id_parts)):
            cur_id = 'T' + id_parts[i]
            res.append("\t".join([cur_id, pos, name]))
        return res

    @staticmethod
    def on_rels_line_pub_tator(line_parts, alias_dict, n_rels):
        rel_type, arg1, arg2 = line_parts[1], line_parts[2], line_parts[3]
        arg1 = "T"+alias_dict[arg1] if arg1 in alias_dict else "T"+arg1
        arg2 = "T"+alias_dict[arg2] if arg2 in alias_dict else "T"+arg2
        assert arg1 != arg2
        return "R"+str(n_rels)+"\t"+rel_type+" "+"Arg1:"+arg1+" Arg2:"+arg2

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
                    ann_lines += self.on_ents_line_pub_tator(rec_parts)
                elif rec_parts[1].isalpha():
                    ann_lines.append(self.on_rels_line_pub_tator(rec_parts, alias_dict, n_rels))
                    n_rels += 1
            elif line == "\n":
                return text, ann_lines
        return text, ann_lines

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

    @staticmethod
    def chunks(l, n):
        res = []
        for i in range(0, len(l), n):
            assert len(l[i:i + n]) == n
            res += [l[i:i + n]]
        return res

    def proc_pub_tator_filterd_file(self, in_file_name):
        pmids = set()
        features = []
        maxlen = 0
        with open(in_file_name, 'r') as infile:
            lines = infile.readlines()
            for i_l, line in enumerate(lines):
                line = line.rstrip().split('\t')
                pmid = line[0]

                if pmid not in pmids:
                    pmids.add(pmid)
                    text = line[1]
                    prs = self.chunks(line[2:], 17)

                    ent2idx = {}
                    train_triples = {}

                    entity_pos = set()
                    for p in prs:
                        es = list(map(int, p[8].split(':')))
                        ed = list(map(int, p[9].split(':')))
                        tpy = p[7]
                        for start, end in zip(es, ed):
                            entity_pos.add((start, end, tpy))

                        es = list(map(int, p[14].split(':')))
                        ed = list(map(int, p[15].split(':')))
                        tpy = p[13]
                        for start, end in zip(es, ed):
                            entity_pos.add((start, end, tpy))

                    sents = [t.split(' ') for t in text.split('|')]
                    new_sents = []
                    sent_map = {}
                    i_t = 0
                    for sent in sents:
                        for token in sent:
                            tokens_wordpiece = self.tokenizer.tokenize(token)
                            for start, end, tpy in list(entity_pos):
                                if i_t == start:
                                    tokens_wordpiece = ["*"] + tokens_wordpiece
                                if i_t + 1 == end:
                                    tokens_wordpiece = tokens_wordpiece + ["*"]
                            sent_map[i_t] = len(new_sents)
                            new_sents.extend(tokens_wordpiece)
                            i_t += 1
                        sent_map[i_t] = len(new_sents)
                    sents = new_sents

                    entity_pos = []

                    for p in prs:
                        if p[0] == "not_include":
                            continue
                        if p[1] == "L2R":
                            h_id, t_id = p[5], p[11]
                            h_start, t_start = p[8], p[14]
                            h_end, t_end = p[9], p[15]
                        else:
                            t_id, h_id = p[5], p[11]
                            t_start, h_start = p[8], p[14]
                            t_end, h_end = p[9], p[15]
                        h_start = map(int, h_start.split(':'))
                        h_end = map(int, h_end.split(':'))
                        t_start = map(int, t_start.split(':'))
                        t_end = map(int, t_end.split(':'))
                        h_start = [sent_map[idx] for idx in h_start]
                        h_end = [sent_map[idx] for idx in h_end]
                        t_start = [sent_map[idx] for idx in t_start]
                        t_end = [sent_map[idx] for idx in t_end]
                        if h_id not in ent2idx:
                            ent2idx[h_id] = len(ent2idx)
                            entity_pos.append(list(zip(h_start, h_end)))
                        if t_id not in ent2idx:
                            ent2idx[t_id] = len(ent2idx)
                            entity_pos.append(list(zip(t_start, t_end)))
                        h_id, t_id = ent2idx[h_id], ent2idx[t_id]

                        r = self.cdr_rel2id[p[0]]
                        if (h_id, t_id) not in train_triples:
                            train_triples[(h_id, t_id)] = [{'relation': r}]
                        else:
                            train_triples[(h_id, t_id)].append({'relation': r})

                    relations, hts = [], []
                    for h, t in train_triples.keys():
                        relation = [0] * len(self.cdr_rel2id)
                        for mention in train_triples[h, t]:
                            relation[mention["relation"]] = 1
                        relations.append(relation)
                        hts.append([h, t])

                maxlen = max(maxlen, len(sents))
                sents = sents[:self.max_seq_len - 2]
                input_ids = self.tokenizer.convert_tokens_to_ids(sents)
                input_ids = [101] + input_ids + [102]

                if len(hts) > 0:
                    feature = {'input_ids': input_ids,
                               'entity_pos': entity_pos,
                               'labels': relations,
                               'hts': hts,
                               'title': pmid,
                               }
                    features.append(feature)
        return features


