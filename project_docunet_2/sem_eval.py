import os
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer

class SemEvalDataProcessor:

    def __init__(self, tokenizer):
        self.max_size = 500
        self.tokenizer = tokenizer

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
        arg1 = "T" + alias_dict[arg1] if arg1 in alias_dict else "T" + arg1
        arg2 = "T" + alias_dict[arg2] if arg2 in alias_dict else "T" + arg2
        assert arg1 != arg2
        return "R" + str(n_rels) + "\t" + rel_type + " " + "Arg1:" + arg1 + " Arg2:" + arg2

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

    @staticmethod
    def on_rel_line_bart(line, rels, allowed_ents):
        line = line.strip()
        idx, meta = line.split('\t')
        rel_name, evid1, evid2 = meta.split(' ')
        evid1, evid2 = evid1.split(':')[1], evid2.split(':')[1]
        if evid1 in allowed_ents and evid2 in allowed_ents:
            rels.append((rel_name, evid1, evid2))

    @staticmethod
    def on_evid_line_bart(line, evid_list):
        line = line.strip()
        line_parts = line.split("\t")
        # ann_idx may be unique or not / same ents may have different names
        ann_idx, loc, name = line_parts[0], line_parts[1], ' '.join(line_parts[2:])
        loc = loc.split(';')[0]
        ent_type, beg, end = loc.split(" ")
        start_num, end_num = min(int(beg), int(end)), max(int(beg), int(end))
        evid_list.append((name, start_num, end_num, ann_idx, ent_type))

    def split_to_fragments(self, text_input, sorted_evids):
        fragments = []
        ents_in_fragments = []
        thre = 0
        prev_ent_end = 0 # for nested / intersected

        cur_fragment, cur_ents, cur_tokens = "", [], 0
        for evid in sorted_evids:
            name, start_num, end_num, ann_idx, ent_type = evid
            p_tokens = self.tokenizer.tokenize(text_input[thre:start_num])
            e_tokens = self.tokenizer.tokenize(text_input[max(start_num, prev_ent_end):end_num])
            if len(e_tokens) == 0:
                assert len(p_tokens) == 0
                cur_ents.append((new_start, new_end, ann_idx))
                continue
            if cur_tokens + len(p_tokens) < self.max_size:
                cur_fragment += self.tokenizer.convert_tokens_to_string(p_tokens) + ' '
                cur_tokens += len(p_tokens)
                thre = start_num

                if cur_tokens + len(e_tokens) < self.max_size:
                    new_start = len(word_tokenize(cur_fragment))
                    cur_fragment += self.tokenizer.convert_tokens_to_string(e_tokens) + ' '
                    cur_tokens += len(e_tokens)
                    thre = end_num
                    prev_ent_end = end_num
                    new_end = len(word_tokenize(cur_fragment))
                    cur_ents.append((new_start, new_end, ann_idx))
                else:
                    fragments.append(cur_fragment)
                    ents_in_fragments.append(cur_ents)
                    cur_fragment, cur_ents, cur_tokens = "", [], 0
            else:
                fragments.append(cur_fragment)
                ents_in_fragments.append(cur_ents)
                cur_fragment, cur_ents, cur_tokens = "", [], 0
        if cur_fragment != '' and len(cur_fragment) > 0:
            fragments.append(cur_fragment)
            ents_in_fragments.append(cur_ents)
        return fragments, ents_in_fragments

    def proc_fragment(self, fragment, ents_in_fragment, ann_input):
        ents = {}
        for e in ents_in_fragment:
            beg, end, lbl = e
            ents[lbl] = (beg, end)
        rels = []
        for l in ann_input:
            if l.startswith('R'):
                self.on_rel_line_bart(l, rels, list(ents.keys()))
        fragment_meta = ""
        if len(rels) == 0:
            return ''
        for rel in rels:
            rel_name, evid1, evid2 = rel
            e1_beg, e1_end = ents[evid1]
            e2_beg, e2_end = ents[evid2]
            rel_stat = '\t' + rel_name + '\t' + str(e1_beg) + '\t' + str(e1_end) + '\t' + evid1 + '\t' + str(e2_beg) + '\t' + str(e2_end) + '\t' + evid2
            fragment_meta += rel_stat
        return fragment + fragment_meta

    def convert_doc_bart(self, ann_input, text_input):
        evids = []
        res = []
        for l in ann_input:
            if l.startswith('T'):
                self.on_evid_line_bart(l, evids)
        sorted_evids = sorted(evids, key=lambda x: (x[1], x[2]))
        fragments, ents_in_fragments = self.split_to_fragments(text_input, sorted_evids)
        for i in range(len(fragments)):
            outp = self.proc_fragment(fragments[i], ents_in_fragments[i], ann_input)
            res.append(outp)
        return res

    def proc_pub_tator_file(self, in_file_name):
        res = []
        with open(in_file_name) as in_file:
            while True:
                text, ann = self.pubTator_to_bart(in_file)
                if text == '':
                    break
                res += self.convert_doc_bart(ann, text)
        return res

    def proc_bart_dir(self, w_dir='.', doc_list=None):
        res = []
        if doc_list:
            documents = doc_list
        else:
            file_names_list = os.listdir(path=w_dir)
            documents = set()
            for file_name in file_names_list:
                doc_id = file_name[:-4]
                documents.add(doc_id)
        for doc in documents:
            print(doc)
            txt, ann = doc + '.txt', doc + '.ann'
            txt_file, ann_file = open(w_dir + '/' + txt), open(w_dir + '/' + ann)
            text_input, ann_input = txt_file.read(), [line.strip() for line in ann_file]
            res += self.convert_doc_bart(ann_input, text_input)
        return res

"""
if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
    cnv = SemEvalDataProcessor(tokenizer)
    train_res = []
    train_res += cnv.proc_pub_tator_file("/home/a-shp/Documents/универ/MSU_AL/ds_for_sem/CDR_DevelopmentSet.PubTator.txt")
    train_res += cnv.proc_pub_tator_file(
        "/home/a-shp/Documents/универ/MSU_AL/ds_for_sem/CDR_TrainingSet.PubTator.txt")
    test_res = cnv.proc_pub_tator_file(
        "/home/a-shp/Documents/универ/MSU_AL/ds_for_sem/CDR_TestSet.PubTator.txt")
    with open('/home/a-shp/Documents/универ/MSU_AL/ds_for_sem/cdr_biobert/SemEval.test.tsv', 'w') as f:
        for line in test_res:
            if line != "":
                f.write(line)
                f.write('\n')
    with open('/home/a-shp/Documents/универ/MSU_AL/ds_for_sem/cdr_biobert/SemEval.train.tsv', 'w') as f:
        for line in train_res:
            if line != "":
                f.write(line)
                f.write('\n')
"""

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    cnv = SemEvalDataProcessor(tokenizer)
    dirs = ['0', '1', '2', '3', '4', '5', '10', '11', '12', '13', '14']
    res_folds = []
    path_start = "/home/a-shp/Documents/универ/MSU_AL/курсовая_маг/ru_datasets/corpus_release/data/cv/"
    for d in dirs:
        print(d)
        print('-----')
        res = cnv.proc_bart_dir(path_start+d)
        res_folds.append(res)

    for i in range(len(dirs)):
        d = dirs[i]
        os.makedirs('/home/a-shp/Documents/универ/MSU_AL/ds_for_sem/cv_base_multilingual/'+d)
        with open('/home/a-shp/Documents/универ/MSU_AL/ds_for_sem/cv_base_multilingual/'+d+'/SemEval.test.tsv', 'w') as f:
            for line in res_folds[i]:
                if line != "":
                    f.write(line)
                    f.write('\n')
        with open('/home/a-shp/Documents/универ/MSU_AL/ds_for_sem/cv_base_multilingual/'+d+'/SemEval.train.tsv', 'w') as f:
            tmp = []
            for j in range(len(res_folds)):
                if i != j:
                    tmp += res_folds[j]
            for line in tmp:
                if line != "":
                    f.write(line)
                    f.write('\n')
