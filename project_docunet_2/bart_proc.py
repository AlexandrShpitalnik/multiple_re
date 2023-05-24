import os

class BartParser:
    def __init__(self):
        pass

    def on_rel_line_bart(self, line, rels, id_aliases):
        line = line.strip()
        idx, meta = line.split('\t')
        rel_name, evid1, evid2 = meta.split(' ')
        if rel_name in ['TO_DETECT', 'TO_STUDY']:
            rel_name = 'TO_DETECT_OR_STUDY'
        rel_name = 'USED_IN' if rel_name == 'CHEMICAL_USED' else rel_name
        rel_name = 'FINDING_OF' if rel_name == 'BIOLOGICAL_PHEN_OF' else rel_name
        rel_name = 'ASSOCIATED_WITH' if rel_name == 'HAS_SYMPTOM' else rel_name
        evid1_id, evid2_id = evid1.split(':')[1], evid2.split(':')[1]
        if rel_name in ['ALTERNATIVE_NAME', 'ABBREVIATION']:
            self.add_to_dict(id_aliases, evid2_id, evid1_id)
            return
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
        evid_list.append((name, start_num, end_num, ann_idx, ent_type))

    def on_alias_line_bart(self, line, aliases):
        line = line.strip()
        names = line.split(" ")
        for name in names[2:]:
            self.add_to_dict(aliases, names[1].lower(), name.lower())

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
            txt, ann = doc + '.txt', doc + '.ann'
            txt_file, ann_file = open(w_dir + '/' + txt), open(w_dir + '/' + ann)
            text_input, ann_input = txt_file.read(), [line.strip() for line in ann_file]
            try:
                doc_res = self.proc_doc_bart(ann_input, text_input)
                res += doc_res
            except:
                pass
        print(self.rels_in_directory, self.rels_found_in_directory)
        return res

