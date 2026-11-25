"""Rough word counts of paper/scireports/main.tex (title, abstract, main text before Methods).
Usage: python3 scripts/scireports/count_words.py paper/scireports/main.tex"""
import re, sys
t = open(sys.argv[1], encoding="utf-8").read()

def clean(s):
    s = re.sub(r"(?<!\\)%.*", "", s)
    s = re.sub(r"\\begin\{(table|figure|align)\}.*?\\end\{\1\}", " ", s, flags=re.S)  # tables, figures, equations
    s = re.sub(r"\\cite\{[^}]*\}", "", s)
    s = re.sub(r"\\(ref|label)\{[^}]*\}", "X", s)
    s = re.sub(r"\$[^$]*\$", "M", s)
    s = re.sub(r"\\(emph|textbf|tbf|paragraph|subsection\*?|subsubsection\*?|section)\{([^}]*)\}", r" \2 ", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    s = re.sub(r"[{}~\\]", " ", s)
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\.,%\u2013']*", s))

title = re.search(r"\\title\[[^\]]*\]\{(.*?)\}\n", t, re.S).group(1)
abstract = re.search(r"\\abstract\{(.*?)\}\n", t, re.S).group(1)
body = t[t.index("\\section{Introduction}"):t.index("\\section{Methods}")]
meth = t[t.index("\\section{Methods}"):t.index("\\section{Acknowledgements}")]
print("title words:", clean(title).__int__() if False else len(re.findall(r"\S+", re.sub(r"--", " ", title))))
print("abstract words:", clean(abstract))
print("main text (Intro+Results+Discussion, no tables/figures/captions):", clean(body))
print("methods:", clean(meth))
