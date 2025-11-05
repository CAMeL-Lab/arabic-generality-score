=================================================================
=================================================================
MADAR Parallel Corpora

Release: version 1.1
Release Date: 18 March 2021 

================================================================
Copyright (c) 2019 Carnegie Mellon University Qatar and New York
University Abu Dhabi. All rights reserved.
=================================================================
=================================================================
Release Update Notes
=================================================================
The following are the main differences from the previous release 
(version 1.0, 8 November 2019):
* Malformed sentence IDs which did not match the original BTEC 
files were fixed.
* 32 malformed sentences from corpus.26 also had to be adjusted.
* Additional columns were added to each file to enhance usability, 
namely:
    - a "split" column linking to the MADAR Shared Task splits, 
    e.g. corpus-6-train and corpus-6-test-corpus-26-dev.
    - a "lang" column containing the 3 letter city code for each 
    dialect, as well as FR, EN, and MSA for French, English, and 
    Modern Standard Arabic respectively.
    - a “sentID.BTEC” column linking to the original BTEC sentences.

=================================================================
Summary
=================================================================
The MADAR corpus is a collection of parallel sentences covering
the dialects of 25 cities from the Arab World, in addition to
English, French, and MSA. The corpus is created by translating
selected sentences from the Basic Traveling Expression Corpus
(BTEC) (Takezawa et al., 2007) to the different dialects.
The exact details on the translation process and source and
target languages are described in Bouamor et al. (2018).

The list of Arab cities covered in the MADAR corpus includes:
Aleppo, Alexandria, Algiers, Amman, Aswan, Baghdad, Basra, Beirut,
Benghazi, Cairo, Damascus, Doha, Fes, Jeddah, Jerusalem, Khartoum,
Mosul, Muscat, Rabat, Riyadh, Salt, Sanaa, Sfax, Tripoli,
and Tunis.

This release contains two datasets:

* Corpus-26: a set of 2,000 BTEC sentences and translated to
all 25 city dialects (each of these sentences has 25 corresponding
parallel translations), in addition to MSA.

* Corpus-6: a set of 12,000 sentences translated to the dialects
of five selected cities: Doha, Beirut, Cairo, Tunis, and Rabat,
in addition to MSA.

Note: We do not provide the English or the French versions of the
corpus because of copyright restrictions.  In order to get access to
the English corpus, you will have to contact NICT and the U-Star
Consortium (Prof. Eiichiro Sumita, eiichiro.sumita@nict.go.jp) to
obtain an end-user agreement that you will have to sign.  The current
contacts for the French version can also be obtained from the U-Star
Consortium.

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
MADAR Corpus
=================================================================

Description of Data:
--------------------

The folder contains 28 tsv files (25 for each city dialect + MSA + 
English (sentID.BTEC only) + French (sentID.BTEC only):

MADAR.Corpus6 - Each of the following 7 tsv files contains a set of 
12,000 entries, 2,000 of which are parallel with MADAR.Corpus26 
linked with sentID.BTEC:
 
  * MADAR.Corpus6.MSA.tsv
  * MADAR.Corpus6.Beirut.tsv
  * MADAR.Corpus6.Cairo.tsv
  * MADAR.Corpus6.Doha.tsv
  * MADAR.Corpus6.Rabat.tsv
  * MADAR.Corpus6.Tunis.tsv
  * MADAR.Corpus6.French.index.tsv (ID only)
  * MADAR.Corpus6.English.index.tsv (ID only)

MADAR.Corpus26 - Each of the following 20 tsv files contains a set 
of 2,000 entries.

  * MADAR.Corpus26.Aleppo
  * MADAR.Corpus26.Alexandria
  * MADAR.Corpus26.Algiers
  * MADAR.Corpus26.Amman
  * MADAR.Corpus26.Aswan
  * MADAR.Corpus26.Baghdad
  * MADAR.Corpus26.Basra
  * MADAR.Corpus26.Benghazi
  * MADAR.Corpus26.Damascus
  * MADAR.Corpus26.Fes
  * MADAR.Corpus26.Jeddah
  * MADAR.Corpus26.Jerusalem
  * MADAR.Corpus26.Khartoum
  * MADAR.Corpus26.Mosul
  * MADAR.Corpus26.Muscat
  * MADAR.Corpus26.Riyadh
  * MADAR.Corpus26.Salt
  * MADAR.Corpus26.Sanaa
  * MADAR.Corpus26.Sfax
  * MADAR.Corpus26.Tripoli

The sentID.BTEC column in each file corresponds to the sentence 
line number in the original English and French BTEC files.  

The files contain the following splits, corresponding to the splits 
in the MADAR Shared Task:

split : count
------------------------------------------------
Corpus-6-train : 9000
Corpus-6-dev : 1000
Corpus-6-test-corpus-26-train : 1600
Corpus-6-test-corpus-26-dev : 200
Corpus-6-test-corpus-26-test : 200

* README.txt
* LICENSE.txt
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

================================================================
Copyright (c) 2019 Carnegie Mellon University Qatar and New York
University Abu Dhabi. All rights reserved.
================================================================

