import json
import numpy as np

def load_ents(ent_file):
    idx2ent = {}
    with open(ent_file + 'ent_ids_1', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            item = line.strip().split('\t')
            if 'zh_en' in ent_file:
                prefix = 'http://zh.dbpedia.org/resource/'
            elif 'ja_en' in ent_file:
                prefix = 'http://ja.dbpedia.org/resource/'
            else:
                prefix = 'http://fr.dbpedia.org/resource/'
            idx = int(item[0])
            ent = item[1].strip().split(prefix)[-1]
            idx2ent[idx] = ent
    with open(ent_file + 'ent_ids_2', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            item = line.strip().split('\t')
            idx = int(item[0])
            ent = item[1].strip().split('http://dbpedia.org/resource/')[-1]
            idx2ent[idx] = ent
    return idx2ent

def load_orig_ents(ent_file):
    if '_en' in ent_file:
        with open(ent_file + 'trans_name.json', 'r', encoding='utf-8') as f:
            trans_name = json.load(f)
        idx2trans, idx2ent, ents1, ents2 = {}, {}, [], []
        for k in trans_name:
            idx2trans[int(k)] = trans_name[k]
            ents1.append(int(k))
        with open(ent_file + 'ent_ids_1', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                item = line.strip().split('\t')
                if 'zh_en' in ent_file:
                    prefix = 'http://zh.dbpedia.org/resource/'
                elif 'ja_en' in ent_file:
                    prefix = 'http://ja.dbpedia.org/resource/'
                else:
                    prefix = 'http://fr.dbpedia.org/resource/'
                idx = int(item[0])
                ent = item[1].strip().split(prefix)[-1]
                ent = ent.replace('_', ' ')
                idx2ent[idx] = ent
        with open(ent_file + 'ent_ids_2', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                item = line.strip().split('\t')
                idx = int(item[0])
                ents2.append(idx)
                ent = item[1].strip().split('http://dbpedia.org/resource/')[-1]
                ent = ent.replace('_', ' ')
                idx2ent[idx] = ent
    else:
        with open(ent_file + 'trans_name_1.json', 'r', encoding='utf-8') as f:
            trans_name_1 = json.load(f)
        with open(ent_file + 'trans_name_2.json', 'r', encoding='utf-8') as f:
            trans_name_2 = json.load(f)
        idx2trans, idx2ent, ents1, ents2 = {}, {}, [], []
        for k in trans_name_1:
            idx2trans[int(k)] = trans_name_1[k]
            ents1.append(int(k))
        for k in trans_name_2:
            idx2trans[int(k)] = trans_name_2[k]
            ents2.append(int(k))
        with open(ent_file + 'ent_ids_1', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                item = line.strip().split('\t')
                idx = int(item[0])
                ent = item[1].strip().split('http://dbpedia.org/resource/')[-1]
                ent = ent.replace('_', ' ')
                idx2ent[idx] = ent
        with open(ent_file + 'ent_ids_2', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                item = line.strip().split('\t')
                if 'en_de' in ent_file:
                    prefix = 'http://de.dbpedia.org/resource/'
                else:
                    prefix = 'http://fr.dbpedia.org/resource/'
                idx = int(item[0])
                ent = item[1].strip().split(prefix)[-1]
                ent = ent.replace('_', ' ')
                idx2ent[idx] = ent

    return idx2trans, idx2ent, ents1, ents2

def load_dwynb_ents(ent_file):
    idx2ent = {}
    ents1, ents2 = [], []
    with open(ent_file + 'ent_ids_1', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            item = line.strip().split('\t')
            idx = int(item[0])
            ent = item[1].rstrip('/').split('/')[-1]
            ent = ent.replace('_', ' ')
            idx2ent[idx] = ent
            ents1.append(idx)
    with open(ent_file + 'ent_ids_2', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            item = line.strip().split('\t')
            idx = int(item[0])
            ents2.append(idx)
            ent = item[1].rstrip('/').split('/')[-1]
            ent = ent.replace('_', ' ')
            idx2ent[idx] = ent
    return idx2ent, ents1, ents2


def load_rels(rel_file):
    idx2rel = {}
    with open(rel_file + 'rel_ids_1', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            item = line.strip().split('\t')
            if 'zh_en' in rel_file:
                prefix = 'http://zh.dbpedia.org/property/'
            elif 'ja_en' in rel_file:
                prefix = 'http://ja.dbpedia.org/property/'
            else:
                prefix = 'http://fr.dbpedia.org/property/'
            idx = int(item[0])
            ent = item[1].strip().split(prefix)[-1]
            idx2rel[idx] = ent
    with open(rel_file + 'rel_ids_2', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            item = line.strip().split('\t')
            idx = int(item[0])
            ent = item[1].strip().split('http://dbpedia.org/property/')[-1]
            idx2rel[idx] = ent
    return idx2rel

def load_triples(triple_file, idx2ents, idx2rels):
    triple_dict = {}
    triples_1 = np.loadtxt(triple_file + 'triples_1', delimiter='\t', dtype=int)
    triples_2 = np.loadtxt(triple_file + 'triples_2', delimiter='\t', dtype=int)
    triples = np.concatenate((triples_1, triples_2), axis=0)
    for triple in triples:
        if triple[0] not in triple_dict:
            triple_dict[triple[0]] = []
        triple_dict[triple[0]].append((idx2ents[triple[0]], idx2rels[triple[1]], idx2ents[triple[2]]))
        if triple[2] not in triple_dict:
            triple_dict[triple[2]] = []
        triple_dict[triple[2]].append((idx2ents[triple[0]], idx2rels[triple[1]], idx2ents[triple[2]]))
    return triple_dict

def load_sft_triples(triple_file, idx2ents, idx2trans, idx2rels):
    orig_triple_dict, trans_triple_dict = {}, {}
    triples_1 = np.loadtxt(triple_file + 'triples_1', delimiter='\t', dtype=int)
    triples_2 = np.loadtxt(triple_file + 'triples_2', delimiter='\t', dtype=int)
    triples = np.concatenate((triples_1, triples_2), axis=0)
    for triple in triples:
        if triple[0] not in orig_triple_dict:
            orig_triple_dict[triple[0]] = []
        orig_triple_dict[triple[0]].append((idx2ents[triple[0]], idx2rels[triple[1]], idx2ents[triple[2]]))
        if triple[2] not in orig_triple_dict:
            orig_triple_dict[triple[2]] = []
        orig_triple_dict[triple[2]].append((idx2ents[triple[0]], idx2rels[triple[1]], idx2ents[triple[2]]))
    for triple in triples:
        if triple[0] in idx2trans:
            if triple[0] not in trans_triple_dict:
                trans_triple_dict[triple[0]] = []
            trans_triple_dict[triple[0]].append((idx2trans[triple[0]], idx2rels[triple[1]], idx2trans[triple[2]]))
        if triple[2] in idx2trans:
            if triple[2] not in trans_triple_dict:
                trans_triple_dict[triple[2]] = []
            trans_triple_dict[triple[2]].append((idx2trans[triple[0]], idx2rels[triple[1]], idx2trans[triple[2]]))
    return orig_triple_dict, trans_triple_dict
