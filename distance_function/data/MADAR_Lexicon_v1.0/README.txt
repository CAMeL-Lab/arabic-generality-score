=================================================================
=================================================================
The MADAR Lexicon

Release: version 1.0
Release Date: 10 November 2022

================================================================
Copyright (c) 2018-2022 Carnegie Mellon University Qatar and New York
University Abu Dhabi. All rights reserved.
=================================================================
=================================================================
Introduction
=================================================================

The MADAR Lexicon is a collection of 1,042* concepts expressed in 25
city dialects totaling 47K entries (with an average of 45 words per
concept, or about 2 words per dialect).  The lexicon is centered around
concept keys, which are triplets of English, French, and Modern Standard
Arabic (MSA), and annotators had to provide words that overlap in word
sense with all three languages.  Concepts were selected from the BTEC
Parallel corpora, and cover 88.0%, 86.4% and 85.5% of the lemma tokens
in the English, French and MSA of each BTEC corpus, respectively. Almost
three-quarters of the concepts are for open classes.  The lexicon also
contains multi-word expressions, and entries for numbers, pronouns,
prepositions, and other closed classes. 

Each dialectal word is presented in its CODA orthography and its CAPHI
phonology (Bouamor et al., 2018; Habash et al., 2018).

The list of Arab cities covered in the MADAR lexicon includes:
Aleppo, Alexandria, Algiers, Amman, Aswan, Baghdad, Basra, Beirut,
Benghazi, Cairo, Damascus, Doha, Fes, Jeddah, Jerusalem, Khartoum,
Mosul, Muscat, Rabat, Riyadh, Salt, Sanaa, Sfax, Tripoli,
and Tunis.

The MADAR Lexicon was created as part of the Multi-Arabic Dialect 
Applications and Resources Project (funded by NPRP 7-290- 1-047 from 
the Qatar National Research Fund (a member of the Qatar Foundation).  
Website: http://madar.camel-lab.com/.

*Different from the number reported in Bouamor et al. (2018) because two
concepts were merged during the quality check effort.

=================================================================
MADAR Lexicon Description
=================================================================
The zipped folder contains the following files:
* MADAR_Lexicon_v1.0/MADAR_Lexicon_v1.0.tsv
* README.txt (this file)
* LICENSE.txt 

The description of the columns of the MADAR_Lexicon_v1.0.tsv file:

1) ID: Entry ID, unique to each entry corresponding to one of 25 cities
        (for 'Country' and 'Region' classification, see ref [3]); List of IDs
        below.
2) Concept_ID: Concept ID, unique to each concept
3) Class: Word class for the concept triplets (MSA, En, Fr), which can
        be 'OpenClass', 'ClosedClass', or 'Mixed' when one or more of the
        triplet lemmas belong to different word classes, or in multi-word
        expressions.
4) English: Concept representative in English
5) French: Concept representative in French
6) MSA: Concept representative in MSA
7) Dialect: Dialect ID (see Dialect ID table below)
8) Example: Optional value to help disambiguate some concepts
9) En_lemma_POS: Lemmatized and tagged (using Stanford POS tagger)
10) Fr_lemma_POS: Lemmatized and tagged (using TreeTagger)
11) MSA_lemma_POS: Lemmatized and tagged (using SAMA v3.1)
12) CODA: Orthographic representation using the Conventional Orthography
        for Dialectal Arabic (see ref [3])
13) Tokenization: CODA, tokenized in D3
14) CAPHI: Phonological representation using the CAMeL Phonetic
        Inventory (see ref [3])
15) Release Status: Marks entries that might need further quality check
16) Release Note: Optional note on "Release Status"

Dialect IDs:

Label:     City
ALE:     Aleppo
ALG:     Algiers
ALX:     Alexandria
AMM:     Amman
ASW:     Aswan
BAG:     Baghdad
BAS:     Basra
BEI:     Beirut
BEN:     Benghazi
CAI:     Cairo
DAM:     Damascus
DOH:     Doha
FES:     Fes
JED:     Jeddah
JER:     Jerusalem
KHA:     Khartoum
MOS:     Mosul
MSA:     Modern
MUS:     Muscat
RAB:     Rabat
RIY:     Riyadh
SAL:     Salt
SAN:     Sana’a
SFX:     Sfax
TRI:     Tripoli
TUN:     Tunis

=================================================================
Citation
=================================================================
When citing this resource, please use:

Bouamor, Houda, Nizar Habash, Mohammad Salameh, Wajdi Zaghouani,
Owen Rambow, Dana Abdulrahim, Ossama Obeid, Salam Khalifa,
Fadhl Eryani, Alexander Erdmann and Kemal Oflazer.
The MADAR Arabic Dialect Corpus and Lexicon.
In Proceedings of the International Conference on Language
Resources and Evaluation (LREC 2018), Miyazaki, Japan, 2018.

@inproceedings{Bouamor:2018:madar,
  Address = {Miyazaki, Japan},
  Author = {Houda Bouamor and Nizar Habash and Mohammad Salameh and
  Wajdi Zaghouani and Owen Rambow and Dana Abdulrahim and Ossama
  Obeid and Salam Khalifa and Fadhl Eryani and Alexander Erdmann
   and Kemal Oflazer},
  Booktitle = {Proceedings of the Language Resources and Evaluation
  Conference (LREC)},
  Title = {{The MADAR {A}rabic Dialect Corpus and Lexicon}},
  Year = {2018}}

=================================================================
References
=================================================================

[1] Bouamor, Houda, Nizar Habash, Mohammad Salameh, Wajdi Zaghouani,
Owen Rambow, Dana Abdulrahim, Ossama Obeid, Salam Khalifa, Fadhl
Eryani, Alexander Erdmann and Kemal Oflazer.  "The MADAR Arabic
Dialect Corpus and Lexicon."  Proceedings of the Eleventh
International Conference on Language Resources and Evaluation (LREC
2018). Miyazaki, Japan, 2018.
http://www.lrec-conf.org/proceedings/lrec2018/pdf/351.pdf

[2] Takezawa, Toshiyuki, Genichiro Kikui, Masahide Mizushima, and
Eiichiro Sumita. 2007. Multilingual Spoken Language Corpus Development
for Communication Research. Computational Linguistics and Chinese
Language Processing, 12(3):303–324.

[3] Habash, Nizar, Fadhl Eryani, Salam Khalifa, Owen Rambow, Dana
Abdulrahim, Alexander Erdmann, Reem Faraj et al. "Unified guidelines and
resources for Arabic dialect orthography." In Proceedings of the
Eleventh International Conference on Language Resources and Evaluation
(LREC 2018). 2018. (See http://coda.camel-lab.com/)

================================================================
Copyright (c) 2018-2022 Carnegie Mellon University Qatar and New York
University Abu Dhabi. All rights reserved.
================================================================
