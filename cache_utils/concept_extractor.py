#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSONL -> JSONL taxonomy extractor using spaCy (POS + dependencies).
Input line:  {"sentence": "..."}
Output line: {"sentence": "...", "Spatial Relationship": [...], "Counting":[...], "Attribute":[...], "Entity":[...]}

Install:
    pip install spacy
    python -m spacy download en_core_web_sm

Run:
    python spacy_taxonomy.py input.jsonl -o output.jsonl
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Set
import pdb
import spacy
from spacy.tokens import Token, Doc

# minimal function words to exclude from spatial anchors
EXCLUDE_ADP = {"of", "to", "for"}  # 功能性介词，非空间
# 常见数量词（非必须；仅作为 POS=ADJ/DET 的补充）
QUANTIFIERS = ["double", "several", "multiple", "many", "few", "both", "dozen", "pair", "couple", "hundred","thousand","million","billion","variety" 'zero', 'none', 'no', 'single', 'a single', 'one', 'only one', 'solo', 'each', 'per', 'every', 'both', 'pair', 'a pair', 'pair of', 'pairs of', 'couple', 'a couple', 'couple of', 'a couple of', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'dozen', 'a dozen', 'half dozen', 'dozen or so', 'several', 'several dozen', 'many', 'numerous', 'multiple', 'various', 'some', 'a few', 'few', 'very few', 'quite a few', 'plenty', 'plenty of', 'lots of', 'a lot of', 'bunch of', 'a bunch of', 'handful', 'handful of', 'a handful of', 'group of', 'a group of', 'crowd of', 'a crowd of', 'cluster of', 'a cluster of', 'stack of', 'a stack of', 'pile of', 'a pile of', 'row of', 'rows of', 'a row of', 'line of', 'lines of', 'series of', 'a series of', 'string of', 'array of', 'set of', 'a set of', 'collection of', 'a collection of', 'assortment of', 'a variety of', 'variety of', 'dozens of', 'scores of', 'hundreds of', 'thousands of', 'tens of', 'hundreds', 'thousands', 'double', 'triple', 'quadruple', 'twice', 'thrice', 'twice as many', 'three times', 'four times', 'five times', 'up to', 'at least', 'at most', 'less than', 'fewer than', 'more than', 'no more than', 'no less than', 'over', 'under', 'about', 'around', 'approximately', 'roughly', 'nearly', 'almost', 'half', 'half of', 'one half', 'quarter', 'a quarter', 'three quarters', 'two thirds', 'one third', 'one per', 'per person', 'per row', 'per table', 'one each', 'several pairs', 'many pairs', 'single pair', 'two or three', 'one or two', 'one more', 'two more', 'several more', 'a couple more', 'countless' ]

SPATIAL_WORDS = ['by','on', 'in', 'under', 'above', 'beside', 'behind', 'left', 'right', 'near', 'front', 'back', 'around', 'between', 'among', 'over', 'below', 'inside', 'inside of', 'outside', 'top', 'bottom', 'center', 'middle', 'edge', 'side', 'next to', 'close to', 'on top', 'in front', 'in back', 'underneath', 'to the left', 'to the right', 'on the left', 'on the right', 'in the middle', 'on the top', 'in the bottom', 'at the back', 'at the front', 'near', 'along', 'across', 'down', 'up', 'over', " out side of", ' outside of',  'through', 'around', 'in front of', 'to the left of', 'to the right of', 'on top of', 'at the back of', 'at the front of', 'in the middle of', 'at the side of', 'on the bottom of', 'at the top of', 'in the bottom of', 'along the edge', 'across the top', 'down the side', 'up the middle', 'over the top', 'through the center', 'around the edge', 'along the line', 'across the middle', 'down the center', 'up the side', 'over the edge', 'through the side', 'behind the center', 'near the side', 'close to the back', 'above the top', 'below the bottom', 'between the two', 'along the back', 'near the front', 'under the top', 'behind the front', 'among the items', 'around the center', 'across the bottom', 'down the back', 'up the front', 'over the middle', 'on the surface', 'in the center', 'under the side', 'on the middle of', 'between the top', 'along the front', 'near the bottom', 'close to the top', 'beside the center', 'behind the bottom', 'in front of the top' ]
def unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for w in items:
        if w not in seen:
            out.append(w); seen.add(w)
    return out

def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        raise RuntimeError(
            "spaCy English model not found. Run: python -m spacy download en_core_web_sm"
        )

def extract_spatial(doc: Doc) -> List[str]:
    """
    抽取空间关系锚点：优先介词/小品词，依存关系为 prep/prt；过滤非空间功能性词。
    """
    anchors: List[str] = []
    # for tok in doc:
    #     if tok.pos_ in ("ADP", "PART") and tok.dep_ in ("prep", "prt"):
    #         # lemma = tok.lemma_.lower()
    #         lemma = str(tok)
    #         flag = 0 
    #         for sw in SPATIAL_WORDS:
    #             if sw in str(doc) and lemma not in EXCLUDE_ADP:
    #                 anchors.append(sw)
    #                 flag = 1
    #         # if lemma and lemma.isalpha() and lemma not in EXCLUDE_ADP and flag == 0:
    #         #     anchors.append(lemma)
    # # 兜底：有些“across/alongside/inside”会被标成 ADV/ADJ，补充常见空间 ADV
    # for tok in doc:
    #     if tok.pos_ in ("ADV", "ADJ"):
    #         # lemma = tok.lemma_.lower()
    #         lemma = str(tok)
    #         for sw in SPATIAL_WORDS:
    #             if sw in str(doc) and lemma not in EXCLUDE_ADP :
    #                 anchors.append(sw)
    #         # if lemma in SPATIAL_WORDS:
    #         #     anchors.append(lemma)
    sws = ''
    for sw in SPATIAL_WORDS:
        if sw in str(doc):
            anchors.append(sw)
    return unique_preserve_order(anchors)

def extract_counting(doc: Doc) -> List[str]:
    """
    抽取计数：NUM 或 like_num，带形态 NumType=Card；再补充常见量词（作为形容词/限定词出现）。
    """
    anchors: List[str] = []
    for tok in doc:
        # lemma = tok.lemma_.lower()
        lemma = str(tok)
        # 纯数字/NUM
        if tok.pos_ == "NUM" or tok.like_num or "NumType=Card" in tok.morph:
            if lemma.isalpha() or tok.text.isdigit():
                anchors.append(lemma if lemma != "-PRON-" else tok.text.lower())
        # 形容词/限定词中的量词（极少词表）
        elif tok.pos_ in ("ADJ", "DET"):
            if lemma in QUANTIFIERS:
                anchors.append(lemma)
    return unique_preserve_order(anchors)

def extract_attribute(doc: Doc, taken: Set[str]) -> List[str]:
    """
    抽取属性：所有形容词（ADJ），排除已占用词；不依赖颜色/材质表，也能覆盖 wooden/steamed 等。
    """
    anchors: List[str] = []
    for tok in doc:
        if tok.pos_ == "ADJ":
            # lemma = tok.lemma_.lower()
            lemma = str(tok)
            if lemma.isalpha() and lemma not in taken:
                anchors.append(lemma)
    return unique_preserve_order(anchors)

def extract_entity(doc: Doc, taken: Set[str]) -> List[str]:
    """
    抽取实体：名词/专有名词（NOUN/PROPN），排除停用词、已占用词和非字母。
    """
    anchors: List[str] = []
    for tok in doc:
        if tok.pos_ in ("NOUN", "PROPN"):
            # 'pots'.lemma_.lower()
            # lemma = tok.lemma_.lower()
            lemma = str(tok)
            if (
                lemma
                and lemma.isalpha()
                and not tok.is_stop
                and lemma not in taken
            ):
                anchors.append(lemma)
    return unique_preserve_order(anchors)

def process_sentence(nlp, sent: str, filename: str) -> Dict:
    doc = nlp(sent)
    spatial = extract_spatial(doc)
    counting = extract_counting(doc)
    taken = set(spatial) | set(counting)
    attribute = extract_attribute(doc, taken)
    taken |= set(attribute)
    entity = extract_entity(doc, taken)
    return {
        "filename": filename,
        "desc": sent,
        "Spatial Relationship": spatial,
        "Counting": counting,
        "Attribute": attribute,
        "Object": entity
    }

def process_jsonl(nlp, in_path: str, out_path: str):
    fin = sys.stdin if in_path == "-" else open(in_path, "r", encoding="utf-8")
    fout = sys.stdout if out_path in (None, "-", "") else open(out_path, "w", encoding="utf-8")
    try:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sent = obj.get("desc", "")
            filename = obj.get("image_file", "")
            # sent = "Two bicycles and a woman walking in front of a shop."
            out = process_sentence(nlp, sent, filename)
            print(json.dumps(out, ensure_ascii=False), file=fout)
    finally:
        if fin is not sys.stdin:
            fin.close()
        if fout is not sys.stdout:
            fout.close()

def main():
    ap = argparse.ArgumentParser(description="Extract taxonomy anchor words via spaCy POS/deps.")
    ap.add_argument("--input", help="Path to input JSONL (use '-' for stdin)")
    ap.add_argument("-o", "--output", default="-", help="Path to output JSONL (default: stdout)")
    args = ap.parse_args()
    nlp = load_nlp()
    process_jsonl(nlp, args.input, args.output)

if __name__ == "__main__":
    main()

#  python cache_utils/concept_extractor.py --input cache_utils/coco2014_train_captions_outputs_simple.jsonl --output cache_utils/test.json
