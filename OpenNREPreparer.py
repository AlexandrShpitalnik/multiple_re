# -*- coding: utf-8 -*-

from collections import defaultdict
from nltk.tokenize import word_tokenize
import nltk
import json
import os

class OpenNREPreparer:

    def __init__(self, lang='ru', with_negatives=True, use_all_rels=False):
        nltk.download('punkt')
        self.with_negatives = with_negatives
        self.use_all_rels = use_all_rels
        if lang == 'ru':
            self.sent_tokenizer = nltk.data.load('tokenizers/punkt/russian.pickle')
        else:
            self.sent_tokenizer = nltk.tokenize.PunktSentenceTokenizer()
        self.lang = lang
        self.rels_found = set()
        self.en_ru_rels_aliases = {'TO_DETECT': 'TO_DETECT_OR_STUDY', 'TO_STUDY': 'TO_DETECT_OR_STUDY',
                                   'AGENT': 'ASSOCIATED_WITH', 'BIOLOGICAL_PHEN_OF': 'FINDING_OF',
                                   'CHEMICAL_USED': 'USED_IN'}
        self.ru_rels_whitelis = {'SUBCLASS_OF', 'MEDICAL_CONDITION', 'PART_OF', 'HAS_CAUSE', 'PHYSIOLOGY_OF',
                                 'ASSOCIATED_WITH', 'FINDING_OF', 'TREATED_USING', 'AFFECTS', 'TO_DETECT_OR_STUDY',
                                 'USED_IN', 'PROCEDURE_PERFORMED', 'VALUE_IS', 'ABBREVIATION', 'ALTERNATIVE_NAME',
                                 'ORIGINS_FROM', 'APPLIED_TO', 'MEMBER_OF', 'PARTICIPANT_IN', 'LOCATED_IN',
                                 'AGE_IS', 'TAKES_PLACE_IN', 'MENTAL_PROCESS_OF', 'PLACE_RESIDES_IN', 'DOSE_IS',
                                 'HAS_ADMINISTRATION_ROUTE', 'DURATION_OF'}
        self.nested_count = 0
        self.procc_count = 0

    @staticmethod
    def on_evid_line_bart(line):
        line = line.strip()
        line_parts = line.split("\t")
        # ann_idx may be unique or not / same ents may have different names
        ann_idx, loc, name = line_parts[0], line_parts[1], ' '.join(line_parts[2:])
        loc = loc.split(';')[0]
        ent_type, beg, end = loc.split(" ")
        start_pos, end_pos = min(int(beg), int(end)), max(int(beg), int(end))
        return name, start_pos, end_pos, ann_idx

    def on_rel_line_bart(self, line):
        line = line.strip()
        idx, meta = line.split('\t')
        rel_name, evid1, evid2 = meta.split(' ')
        if self.lang == 'eng' and rel_name in self.en_ru_rels_aliases:
            rel_name = self.en_ru_rels_aliases[rel_name]
        if (rel_name in self.ru_rels_whitelis) or self.use_all_rels:
            self.rels_found.add(rel_name)
            evid1_id, evid2_id = evid1.split(':')[1], evid2.split(':')[1]
            return idx, evid1_id, evid2_id, rel_name
        return None

    def validate_sent_rels(self, spans, evids, rels):
        spans_with_rels = defaultdict(list)
        for rel_id in rels.keys():
            e1, e2, _ = rels[rel_id][0]
            e1_start, e1_end, _ = evids[e1]
            e2_start, e2_end, _ = evids[e2]
            if e1_end < e2_start or e1_start > e2_end: 
                window_start, window_end = min(e1_start, e2_start, e1_end, e2_end), max(e1_start, e2_start, e1_end, e2_end)
                for span_start, span_end in spans:
                    if span_start <= window_start and span_end >= window_end:
                        spans_with_rels[(span_start, span_end)].append(rel_id)
                        self.procc_count += 1
            else:
                self.nested_count += 1
        return spans_with_rels

    def generate_negatives(self, rels, rels_in_span):
        span_evid_ids = set()
        true_rels = set()
        for cur_rel in rels_in_span:
            e1_id, e2_id, _ = rels[cur_rel][0]
            span_evid_ids.add(e1_id)
            span_evid_ids.add(e2_id)
            true_rels.add((e1_id, e2_id))
        neg_rels = [(i, j, "NA") for i in span_evid_ids for j in span_evid_ids if (i != j and (i, j) not in true_rels)]
        return {"N"+str(i): [neg_rels[i]] for i in range(len(neg_rels))}

    def convert_sentence(self, text, span, evids, rels, rels_in_span):
        cur_sent_unmarked = text[span[0]:span[1]]
        cur_sent_unmarked_tokens = word_tokenize(cur_sent_unmarked)
        offset = span[0]
        results = []
        if self.with_negatives:
            rels_with_negatives = self.generate_negatives(rels, rels_in_span)
            rels_in_span_with_negatives = rels_in_span + list(rels_with_negatives.keys())
            rels_with_negatives.update(rels)
        else:
            rels_in_span_with_negatives = rels_in_span
            rels_with_negatives = rels

        for cur_rel in rels_in_span_with_negatives:
            e1_id, e2_id, rel_name = rels_with_negatives[cur_rel][0]
            e1_start, e1_end, _ = evids[e1_id]
            e2_start, e2_end, _ = evids[e2_id]
            e1_start -= offset
            e1_end -= offset
            e2_start -= offset
            e2_end -= offset
            if e1_start < e2_start:
                cur_sent = cur_sent_unmarked[:e2_end] + '#' + cur_sent_unmarked[e2_end:]
                cur_sent = cur_sent[:e2_start] + '#' + cur_sent[e2_start:]
                cur_sent = cur_sent[:e1_end] + '@' + cur_sent[e1_end:]
                cur_sent = cur_sent[:e1_start] + '@' + cur_sent[e1_start:]
            else:
                cur_sent = cur_sent_unmarked[:e1_end] + '@' + cur_sent_unmarked[e1_end:]
                cur_sent = cur_sent[:e1_start] + '@' + cur_sent[e1_start:]
                cur_sent = cur_sent[:e2_end] + '#' + cur_sent[e2_end:]
                cur_sent = cur_sent[:e2_start] + '#' + cur_sent[e2_start:]
            tokens = word_tokenize(cur_sent)
            skipped_tokens = 0
            e1_tokens_pos, e2_tokens_pos = [], []
            for token_cnt, token in enumerate(tokens):
                if token == "@":
                    e1_tokens_pos.append(token_cnt - skipped_tokens)
                    skipped_tokens += 1
                elif token == "#":
                    e2_tokens_pos.append(token_cnt - skipped_tokens)
                    skipped_tokens += 1
            e1_name = " ".join(cur_sent_unmarked_tokens[e1_tokens_pos[0]: e1_tokens_pos[1]])
            e2_name = " ".join(cur_sent_unmarked_tokens[e2_tokens_pos[0]: e2_tokens_pos[1]])
            e1_info = {"name": e1_name, "id": e1_id, "pos": e1_tokens_pos}
            e2_info = {"name": e2_name, "id": e2_id, "pos": e2_tokens_pos}
            results.append(json.dumps({"token": cur_sent_unmarked_tokens, "h": e1_info, "t": e2_info,
                                       "relation": rel_name})+'\n')
        return results

    def process_doc(self, working_dir, doc_id):
        with open(working_dir+'/'+doc_id+'.ann') as f:
            ann_input = [line for line in f]
        with open(working_dir+'/'+doc_id+'.txt') as f:
            text = f.read()
        sents_spans = [span for span in self.sent_tokenizer.span_tokenize(text)]
        evids = {}
        rels = defaultdict(list)
        for line in ann_input:
            if line[0] == "T":
                name, start_pos, end_pos, ann_idx = self.on_evid_line_bart(line)
                evids[ann_idx] = (start_pos, end_pos, name)
            elif line[0] == "R":
                rel_info = self.on_rel_line_bart(line)
                if rel_info:
                    rel_id, e1, e2, rel_name = rel_info
                    rels[rel_id].append((e1, e2, rel_name))
        spans_with_rels = self.validate_sent_rels(sents_spans, evids, rels)
        res = []
        for span in sents_spans:
            if span in spans_with_rels:
                res += self.convert_sentence(text, span, evids, rels, spans_with_rels[span])
        return res

    def make_rels_idxs(self):
        return json.dumps({rel_name: i for i, rel_name in enumerate(self.rels_found)})


if __name__ == '__main__':
    preparer = OpenNREPreparer(lang='eng')

    res = []
    for root, dirs, files in os.walk('../v0/med_eng'):
        for filename in files:
            if filename.endswith('.ann'):
                res += preparer.process_doc('../v0/med_eng', filename[:-4])

    sent_tokenizer = nltk.data.load('tokenizers/punkt/russian.pickle')
    cdw = ''
    preparer = OpenNREPreparer()
    for root, dirs, files in os.walk("simple_test"):
        for filename in files:
            print(filename)
            if filename.endswith('.ann'):
                res = preparer.process_doc("simple_test", filename[:-4])
                with open("simple_test/res.txt", "w") as f:
                    for record in res:
                        f.write(record)
                with open("simple_test/res_rels.txt", 'w') as f:
                    f.write(preparer.make_rels_idxs())
