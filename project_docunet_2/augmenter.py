import random
from transformers import AutoTokenizer

class AugmentingTokenizer:
    def __init__(self, name, p = 0.5):
        self.base_tokenizer = AutoTokenizer.from_pretrained(name)
        self.aug_proba  = p
        self.cls_token_id = self.base_tokenizer.cls_token_id
        self.sep_token_id = self.base_tokenizer.sep_token_id

    def subword_augment(self, word, n_subparts=3):
        segment_size = len(word) // n_subparts
        new_tokens = []
        for i in range(n_subparts):
            if i != n_subparts - 1:
                sw = word[i * segment_size:(i + 1) * segment_size]
            else:
                sw = word[i * segment_size:]
            st = self.base_tokenizer.tokenize(sw)
            if i != 0:
                st[0] = "##" + st[0]
            new_tokens += st
        return new_tokens

    def tokenize(self, inp):
        inp = self.base_tokenizer.convert_tokens_to_string(self.base_tokenizer.tokenize(inp))
        res = []
        for word in inp.split(' '):
            res += self.tokenize_aug(word)
        return res

    def tokenize_aug(self, word, n_subparts=3):
        p = random.uniform(0, 1)
        if word.isalpha() and len(word) >= 3 and p < self.aug_proba:
            r = self.subword_augment(word, n_subparts)
            return r
        else:
            return self.base_tokenizer.tokenize(word)

    def convert_ids_to_tokens(self, L):
        if type(L) == list:
            return [self.base_tokenizer.convert_ids_to_tokens(t) for t in L]
        else:
            return self.base_tokenizer.convert_ids_to_tokens(L)

    def convert_tokens_to_ids(self, L):
        if type(L) == list:
            return [self.base_tokenizer.convert_tokens_to_ids(t) for t in L]
        else:
            return self.base_tokenizer.convert_tokens_to_ids(L)



"""
class Augmenter():

    def __init__(self, tokenizer):
        self.overall_ratio = None
        self.n_tokens = None
        self.n_cs_tokens = None
        self.dict_languages = None
        self.code_switch_ratio = None
        self.lang2dict = None
        self.tokenizer = tokenizer

    def switch_token(self, token):


    def tokenize_token(self, token, switch_text=False, enable_code_switch=False, enable_bpe_switch=False,
                       enable_bpe_sampling=False):
        switch_token = random.random() <= self.overall_ratio
        is_switched = False
        self.n_tokens += 1
        if enable_code_switch and switch_text and switch_token and random.random() <= self.code_switch_ratio:
            lang = self.dict_languages[random.randint(0, len(self.dict_languages) - 1)]
            if token.lower() in self.lang2dict[lang]:
                self.n_cs_tokens += 1
                token = self.lang2dict[lang][token.lower()][
                    random.randint(0, len(self.lang2dict[lang][token.lower()]) - 1)]
                is_switched = True

    def tokenize_token_full(self, token, switch_text=False, enable_code_switch=False, enable_bpe_switch=False,
                       enable_bpe_sampling=False):
        switch_token = random.random() <= self.overall_ratio
        is_switched = False
        self.n_tokens += 1
        if enable_code_switch and switch_text and switch_token and random.random() <= self.code_switch_ratio:
            lang = self.dict_languages[random.randint(0, len(self.dict_languages) - 1)]
            if token.lower() in self.lang2dict[lang]:
                self.n_cs_tokens += 1
                token = self.lang2dict[lang][token.lower()][
                    random.randint(0, len(self.lang2dict[lang][token.lower()]) - 1)]
                is_switched = True

        if enable_bpe_switch and switch_text and switch_token and random.random() <= self.bpe_switch_ratio:
            lang = self.tokenizer_languages[random.randint(0, len(self.tokenizer_languages) - 1)]
            tokenizer = self.lang2tokenizer[lang]
            is_switched = True
        else:
            tokenizer = self.tokenizer

        if enable_bpe_sampling and switch_text and switch_token and random.random() <= self.bpe_sampling_ratio:
            sub_tokens = tokenizer.tokenize(token, nbest_size=self.sampling_nbest_size,
                                            alpha=self.sampling_alpha)
            is_switched = True
        else:
            sub_tokens = tokenizer.tokenize(token)

        return sub_tokens, is_switched

class TextAugmenter:

    augment text data for input batch:
    - with word(tokens) dict
    - replace N ratio of tokens with it's translation
        - inside / outside / both input tokens
        
    def __init__(self, tokenizer):
        self.word_dict = None
        self.tokenizer = tokenizer

    def load_dict(self, word_dict):
        self.word_dict = word_dict

    def convert(self, batch, replacement_ratio=0):
        for lang in dict_languages:
            # dict_path = os.path.join(self.dict_dir, "{}2.txt".format(lang))
            dict_path = os.path.join(self.dict_dir, "en-{}.txt".format(lang))
            if not os.path.exists(dict_path):
                logger.info("dictionary en-{} doesn't exist.".format(lang))
                continue
            self.dict_languages.append(lang)
            logger.info("reading dictionary from {}".format(dict_path))
            with open(dict_path, "r", encoding="utf-8") as reader:
                raw = reader.readlines()
            self.lang2dict[lang] = {}
            for line in raw:
                line = line.strip()
                try:
                    src, tgt = line.split("\t")
                except:
                    src, tgt = line.split(" ")
                if src not in self.lang2dict[lang]:
                    self.lang2dict[lang][src] = [tgt]
                else:
                    self.lang2dict[lang][src].append(tgt)

    def word_piece_token_augmenter(self, word, n_subparts=3):
        segment_size = len(word) // n_subparts
        new_tokens = []
        for i in range(n_subparts):
            if i != n_subparts - 1:
                sw = word[i * segment_size:(i + 1) * segment_size]
            else:
                sw = word[i * segment_size:]
            st = self.tokenizer.tokenize(sw)
            if i != 0:
                st[0] = "##" + st[0]
            new_tokens += st
        return new_tokens
    
"""