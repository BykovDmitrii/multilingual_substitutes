import nltk
import spacy
import pickle
import argparse
import time, random
import pymorphy2
from collections import defaultdict
from datasets import load_dataset
from deep_translator import PonsTranslator
from transformers import XLMRobertaTokenizer
from textblob import Word
from functools import lru_cache
from word2word import Word2word

PYMORPHY = pymorphy2.MorphAnalyzer()

TOKENIZER = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")

def encode(s):
    return TOKENIZER([s], return_tensors='pt', padding=True, truncation=True)['input_ids'][0]

MASK_ID = encode("<mask>")[1]
PAD_ID = encode("<pad>")[1]


def tokenize(word):
    return nltk.word_tokenize(word)


def get_translations_pons(words, src_lang, tgt_lang, remove_multi_word=True):
    translator = PonsTranslator(src_lang, tgt_lang)
    NOT_TRANSLATED, TRANSLATION_QUERIES = 0, 0
    unique_words = set(words)
    TRANSLATION_QUERIES += len(unique_words)
    lemma2translations = {}
    for word in unique_words:
        try:
            translations = translator.translate(word, return_all=True)
            #print(word, translations)
        except Exception as ex:
            #print(ex)
            translations = []
            NOT_TRANSLATED += 1
        if remove_multi_word:
            translations = list(set([
                translation.strip() for translation in translations if len(translation.strip().split()) == 1]))
        lemma2translations[word] = translations
    return lemma2translations, TRANSLATION_QUERIES, NOT_TRANSLATED


def get_translations_w2w(words, src_lang: str = "en", dst_lang: str = "es", pos_tags=None):
    en2es = Word2word(src_lang, dst_lang)
    not_translated = 0
    unique_words = set(words)
    lemma2translations = {}
    for word in unique_words:
        try:
            translations = en2es(word)
        except KeyError:
            translations = []
            not_translated += 1
        lemma2translations[word] = translations
    return lemma2translations, len(unique_words) - not_translated, not_translated 


def get_data_by_num_tokens(data, word_translations, inserted_pattern, lemmatize_src, lemmatize_tgt, src_lang, tgt_lang, masked_target_percent=0):
    prepared_data_by_num_target_tokens = defaultdict(list)
    counter = 0
    start_time = time.time()
    for row_num, r in enumerate(data):
        src_tokenized_as_list = tokenize(r['translation'][src_lang])
        src_tokenized = " ".join(src_tokenized_as_list)
        tgt = tokenize(r['translation'][tgt_lang].lower())
        tgt_lem = lemmatize_tgt(r['translation'][tgt_lang].lower())
        for w in src_tokenized_as_list:
            if src_tokenized.count(w) == 1:
                translations = []
                if w in word_translations: 
                    translations = translations + word_translations[w]
                if w.lower() in word_translations:
                    translations = translations + word_translations[w.lower()]
                l = lemmatize_src(w.lower())
                if l in word_translations:
                    translations = translations + word_translations[l]
                translations = list(set(translations))
                for tr in translations:
                    if tr.lower() in tgt or tr.lower() in tgt_lem:
                        y = encode(tr.lower())[1:-1]
                        tr_len = len(y)
                        if tr_len > 3:
                            continue
                        if random.random() < 1 - masked_target_percent:
                            len_t = 0
                            x = encode(src_tokenized.replace(w, inserted_pattern.format(target=w, masks="<mask>" * 3)))
                        else:
                            len_t = len(encode(w)[1:-1])
                            x = encode(src_tokenized.replace(w, inserted_pattern.format(target="<mask>" * len_t, masks="<mask>" * 3)))
                        len_x = len(x)
                        mask_counter = 0
                        for i, el in list(enumerate(x)):
                            if el == MASK_ID and len_t > mask_counter:
                                mask_counter += 1
                                continue
                            if el == MASK_ID:
                                for j in range(3):
                                    mask_counter += 1
                                    el = y[j] if j < tr_len else PAD_ID
                                    if j < tr_len or random.random() < 0.1:
                                    	prepared_data_by_num_target_tokens[(1, len_x)].append((x, el, i + j))
                                    x = x.clone()
                                    x[i+j] = el

                                counter += 1
                                if counter % 1000 == 0:
                                    print(counter, row_num, time.time() - start_time)
                                    print(src_tokenized, inserted_pattern, w, tr, x, TOKENIZER.decode(x))
                                break
                        if mask_counter != len_t + 3:
                            raise Exception("Strange number of masks")
    print("Finished prepare data Total data {} Total time {}".format(counter, time.time() - start_time))
    return prepared_data_by_num_target_tokens


@lru_cache(maxsize=100000)
def lematize_english(word):
    return Word(word).lemmatize()


def lemmatize_russian(sent):
    lemmatized_sent = []
    for word in tokenize(sent.lower()):
        for lemma in set([i.normal_form for i in PYMORPHY.parse(word)]):
            lemmatized_sent.append(lemma)
    return lemmatized_sent


def lematize_with_spacy(lemmatizer, sent):
    result = []
    for token in lemmatizer(sent):
        result.append(token.lemma_)
    return result


def get_word_translations(dataset, source_lang, target_lang):
    words = set()
    for row in dataset:
        for word in tokenize(row['translation']['en']):
            words.add(word)
    print("words ", list(words)[:20])
    pons = {}
    try:
        pons = get_translations_pons(words, source_lang, target_lang)
    except Exception as ex:
        pass
        #print(ex)
    cntr = 0
    if len(pons) > 0:
        for key, val in pons[0].items():
            cntr += len(val)
    if cntr < 100:
        w2w = get_translations_w2w(words, source_lang, target_lang)
        return w2w
    return pons



if __name__ == "__main__":
    source_lang = "en"
    #target_lang = "de"
    parser = argparse.ArgumentParser()
    parser.add_argument('target_lang')
    args = parser.parse_args()
    target_lang = args.target_lang
    masked_target_percent = 0.1
    valid_len = 1000
    lang_full_name = ""
    for row in open("languages_spacy", "r"): 
        if row.split()[1] == target_lang:
            lang_full_name = row.split()[0]
    if lang_full_name != "":
        inserted_pattern = "{target} (" + lang_full_name + ": {masks})"
        if target_lang != "zh":
            spacy_lemmatizer = spacy.load("{}_core_news_sm".format(target_lang))
            lemmatize_function = lambda x: lematize_with_spacy(spacy_lemmatizer, x)
        else:
            spacy_lemmatizer = spacy.load("{}_core_web_sm".format(target_lang))
            lemmatize_function = lambda x: lematize_with_spacy(spacy_lemmatizer, x)
    else:
        inserted_pattern = "{target} [{masks}]"
        lemmatize_function = lambda x: x
    print("started loading dataset")
    try:
        dataset = load_dataset("tatoeba", lang1=source_lang, lang2=target_lang)
    except:
        dataset = load_dataset("tatoeba", lang2=source_lang, lang1=target_lang)

    filename = "{source_lang}_{target_lang}_{masked_target_percent}_{inserted_pattern}_{valid_len}".format(
        source_lang=source_lang,
        target_lang=target_lang,
        masked_target_percent=masked_target_percent,
        inserted_pattern=inserted_pattern,
        valid_len=valid_len)
    dataset = [r for r in dataset['train']]
    print("start word_translations")
    word_translations, _, _ = get_word_translations(dataset, source_lang, target_lang)
    print(word_translations)
    print("All words translated")

    prepared_data_by_num_target_tokens = get_data_by_num_tokens(dataset[:-valid_len], word_translations, inserted_pattern, lematize_english, lemmatize_function, source_lang, target_lang, masked_target_percent)
    with open("train_"+filename, "wb") as f:
        pickle.dump(prepared_data_by_num_target_tokens, f)
    prepared_data_by_num_target_tokens = get_data_by_num_tokens(dataset[-valid_len:], word_translations, inserted_pattern, lematize_english, lemmatize_function, source_lang, target_lang, masked_target_percent)
    with open("valid_"+filename, "wb") as f:
        pickle.dump(prepared_data_by_num_target_tokens, f)
