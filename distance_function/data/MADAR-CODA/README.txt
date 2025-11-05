=================================================================
=================================================================
MADAR CODA Corpus

Release: version 1.0
Release Date: 18 March 2021 

================================================================
Copyright (c) 2021 New York University Abu Dhabi. All rights 
reserved.
=================================================================
=================================================================

Summary
=================================================================
The MADAR CODA Corpus contains 10,000 sentences corrected using 
the Conventional Orthography for Dialectal Arabic (CODA), along 
with their original raw form.  The sentences were taken from the 
5 Arab city dialects (Beirut, Cairo, Doha, Rabat, and Tunis) that 
make up MADAR.Corpus6, a subset of the MADAR Parallel Corpora. 
The corpus was created by a human annotator with the aid of a 
bootstrapping technique, and validated manually.  More details 
are provided in Eryani et al. (2020) (see references).

This release contains two directories, train and test, each of 
which contains 5 tsv files corresponding to each city dialect.  
Each tsv contains 1,000 sentences in raw and CODA form.  To ease 
usability between the different corpora, we follow MADAR Parallel 
Corpora by including the following additional columns in each tsv: 
  1) a sentID.BTEC column, corresponding to the sentence line number 
  in the original English and French BTEC files; 
  2) a splits column, corresponding to the splits in the MADAR 
  Shared Task; 
  3) a dialect column containing the 3 letter code of each city.

When citing this resource, please use:

Eryani, Fadhl, Nizar Habash, Houda Bouamor, and Salam Khalifa. 
"A spelling correction corpus for multiple Arabic dialects." In 
Proceedings of The 12th Language Resources and Evaluation 
Conference, pp. 4130-4138. 2020.

@inproceedings{Eryani:2020:spelling,
  title={A Spelling Correction Corpus for Multiple {A}rabic 
  Dialects},
  author={Eryani, Fadhl and Nizar Habash and Houda Bouamor and 
  Salam Khalifa},
  booktitle={Proceedings of The 12th Language Resources and 
  Evaluation Conference},
  pages={4130--4138},
  year={2020}
}

=================================================================
References
=================================================================

[1] Eryani, Fadhl, Nizar Habash, Houda Bouamor, and Salam Khalifa. 
"A spelling correction corpus for multiple Arabic dialects." In 
Proceedings of The 12th Language Resources and Evaluation 
Conference, pp. 4130-4138. 2020.

[2] Bouamor, Houda, Nizar Habash, Mohammad Salameh, Wajdi Zaghouani,
Owen Rambow, Dana Abdulrahim, Ossama Obeid, Salam Khalifa, Fadhl
Eryani, Alexander Erdmann and Kemal Oflazer.  "The MADAR Arabic
Dialect Corpus and Lexicon."  Proceedings of the Eleventh
International Conference on Language Resources and Evaluation (LREC
2018). Miyazaki, Japan, 2018.
http://www.lrec-conf.org/proceedings/lrec2018/pdf/351.pdf

[3] Takezawa, Toshiyuki, Genichiro Kikui, Masahide Mizushima, and
Eiichiro Sumita. 2007. Multilingual Spoken Language Corpus Development
for Communication Research. Computational Linguistics and Chinese
Language Processing, 12(3):303–324.

================================================================
Copyright (c) 2021 New York University Abu Dhabi. All rights 
reserved.
================================================================

