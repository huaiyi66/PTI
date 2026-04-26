#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Taxonomy anchor-word extractor (single-word only).

Input (JSONL): each line is {"sentence": "..."}.
Output (JSONL): {"sentence": "...", "Spatial Relationship": [...], "Counting": [...], "Attribute": [...], "Entity": [...]}

Rules:
- Anchor words must appear in the sentence (we output in lowercase).
- Only single words (no multi-word phrases).
- Per-category lists have no duplicates (order preserved).
- No external dependencies.
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Set

# -----------------------------
# Lexicons (single-word only)
# -----------------------------

SPATIAL_WORDS: Set[str] = {
    # spatial prepositions / adpositions (single-word; exclude "to", "of")
    "on", "in", "inside", "outside", "under", "over", "above", "below",
    "between", "behind", "near", "beside", "next", "across", "around",
    "through", "against", "within", "beyond", "beneath", "underneath",
    "atop", "by", "along", "down", "up", "before", "after", "beside", "above"
}

COUNTING_WORDS: Set[str] = {
    # cardinal / quantifier words
    "zero","one","two","three","four","five","six","seven","eight","nine","ten",
    "eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen",
    "eighteen","nineteen","twenty","thirty","forty","fifty","sixty","seventy",
    "eighty","ninety","hundred","thousand","million","billion",
    # common quantifiers explicitly used in自然描述
    "dozen","couple","pair","several","multiple","many","few","both"
}

ATTRIBUTE_WORDS: Set[str] = {
    # colors
    "red","blue","green","yellow","orange","purple","violet","indigo","black",
    "white","gray","grey","brown","pink","cyan","magenta","beige","gold","silver",
    # sizes & shapes & typical attributes
    "small","large","big","little","tall","short","wide","narrow","heavy","light",
    "old","young","new","beautiful","pretty","ugly","clean","dirty","bright","dark",
    "hot","cold","warm","cool","fast","slow","round","square","wooden","metal",
    "plastic","sharp","blunt","soft","hard","smooth","rough","fresh","steamed",
    "cooked","raw","ripe","nutritious","appealing","spicy","sweet","sour","salty"
}

# Stopwords and non-entity filters
STOPWORDS: Set[str] = {
    "the","a","an","and","or","but","if","then","else","when","while","as","of",
    "at","for","from","with","without","into","onto","about","than","that","this",
    "these","those","is","are","was","were","be","been","being","to","in","on",
    "by","over","under","it","its","their","his","her","our","your","my","mine",
    "they","them","he","she","we","you","i","there","here","also","both","all",
    "so","very","more","most","such","each","every","any","some","no","not",
    "which","who","whom","whose","what","where","when","why","how"
}

# common verb-like tokens to exclude from Entities
NON_ENTITY_VERB_LIKE: Set[str] = {
    "shows","show","showing","features","feature","featuring","work","working",
    "use","uses","using","make","makes","making","prepare","prepares","preparing",
    "cook","cooks","cooking","cut","cuts","cutting","chop","chops","chopping",
    "mix","mixes","mixing","serve","serves","serving","clean","cleans","cleaning",
    "arrange","arranges","arranging","position","positions","positioned",
    "appear","appears","appearing","designed","design","designs","designed"
}

# -----------------------------
# Tokenization / helpers
# -----------------------------

PUNCT_RE = re.compile(r"[^\w\s]+")

def tokenize(text: str) -> List[str]:
    # normalize to lowercase, remove punctuation (split hyphenated as separate words)
    t = text.lower().replace("-", " ")
    t = PUNCT_RE.sub(" ", t)
    toks = [tok for tok in t.split() if tok]
    return toks

def unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for w in items:
        if w not in seen:
            out.append(w)
            seen.add(w)
    return out

def is_number_token(tok: str) -> bool:
    # digit forms like 3, 12
    return tok.isdigit()

def extract_spatial(tokens: List[str]) -> List[str]:
    return unique_preserve_order([t for t in tokens if t in SPATIAL_WORDS])

def extract_counting(tokens: List[str]) -> List[str]:
    res = []
    for t in tokens:
        if is_number_token(t) or t in COUNTING_WORDS:
            res.append(t)
    return unique_preserve_order(res)

def extract_attribute(tokens: List[str]) -> List[str]:
    return unique_preserve_order([t for t in tokens if t in ATTRIBUTE_WORDS])

def extract_entity(tokens: List[str], already_taken: Set[str]) -> List[str]:
    """
    Heuristic entity picker:
    - word is alphabetic
    - not stopword, not in other categories, not verb/adverb-like
    - avoid common suffixes for non-nouns (ing, ly)
    """
    res = []
    for t in tokens:
        if not t.isalpha():
            continue
        if t in already_taken:
            continue
        if t in STOPWORDS:
            continue
        if t in NON_ENTITY_VERB_LIKE:
            continue
        if t.endswith("ing") or t.endswith("ly"):
            continue
        # looks like a noun-ish candidate
        res.append(t)
    return unique_preserve_order(res)

# -----------------------------
# Core processing
# -----------------------------

def process_sentence(sent: str) -> Dict:
    tokens = tokenize(sent)
    spatial = extract_spatial(tokens)
    counting = extract_counting(tokens)
    attribute = extract_attribute(tokens)

    taken = set(spatial) | set(counting) | set(attribute)
    entity = extract_entity(tokens, taken)

    return {
        "value": sent,
        "Spatial Relationship": spatial,
        "Counting": counting,
        "Attribute": attribute,
        "Entity": entity
    }

def process_jsonl(in_path: str, out_path: str = None) -> None:
    fin = sys.stdin if in_path == "-" else open(in_path, "r", encoding="utf-8")
    fout = sys.stdout if not out_path or out_path == "-" else open(out_path, "w", encoding="utf-8")
    try:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sent = obj.get("value", "")
            out_obj = process_sentence(sent)
            print(json.dumps(out_obj, ensure_ascii=False), file=fout)
    finally:
        if fin is not sys.stdin:
            fin.close()
        if fout is not sys.stdout:
            fout.close()

# -----------------------------
# CLI
# -----------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Extract single-word taxonomy anchors from JSONL sentences.")
    ap.add_argument("--input", help="Path to input JSONL (use '-' for stdin)")
    ap.add_argument("-o", "--output", default="-", help="Path to output JSONL (default: stdout)")
    args = ap.parse_args()
    process_jsonl(args.input, args.output)
