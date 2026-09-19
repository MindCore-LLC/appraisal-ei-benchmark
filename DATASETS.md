# External datasets

This file is the provenance record for third-party corpora used by the
Appraisal-EI Benchmark and the MindCore POC eval pipeline. Raw corpora are
fetched from upstream sources (`mindcore-poc/scripts/fetch_corpora.py`) and
converted (`mindcore-poc/scripts/prepare_corpora.py`); we do not claim
authorship of any of them. Each entry lists what we use it for, access
terms as we understand them, and the required citation.

## Benchmark stimuli (committed to this repo)

### crowd-enVENT (envent_{train,val,test}.jsonl — MEASURED split)

- **What**: 1,200 event descriptions, each rated by 5 independent readers
  reconstructing the experiencer's appraisal on 21 variables (6,000 reader
  annotations over the validation subset of a 6,600-item corpus).
- **How we use it**: `stimuli/text/envent_*.jsonl`. Reader-consensus ratings
  mapped onto our 17-dim schema; these rows carry `source: "human"` and
  produce `status: "measured"` results. `anticipated_emotion` is
  author-grounded (writer's stated emotion x writer's intensity).
- **Access**: free download, citation required; the corpus consent states
  the data is made publicly available in anonymised form.
- **Cite**: Troiano, E., Oberlander, L.A.M., & Klinger, R. (2023).
  Dimensional Modeling of Emotions in Text with Appraisal Theories: Corpus
  Creation, Annotation Reliability, and Prediction. *Computational
  Linguistics* 49(1). doi:10.1162/coli_a_00461
- **Source**: https://www.romanklinger.de/data-sets/crowd-enVent2023.zip

### Dimension mapping (enVENT 21 vars → our 17 dims)

enVENT items are 1-5 Likert ("Not at all" to "Extremely"); our schema is
-3..+3. `L(x) = (x-3)*1.5`; inverted items use `L(6-x)`. Composites average
the mapped components. Verified against the published annotation
questionnaires (`questionnaires/` in the corpus zip).

| Our dim | enVENT variable(s) | Item wording (validation form) |
|---|---|---|
| pleasantness | pleasantness - unpleasantness | "The event was pleasant/unpleasant for the experiencer." |
| goal_relevance | goal_relevance | "expected the event to have important consequences" |
| goal_congruence | goal_support | "expected positive consequences" |
| certainty | predict_event + predict_conseq | "could have predicted the occurrence/consequences" |
| control | self_control | "was able to influence what was going on" |
| responsibility | self_responsblt | "caused by the experiencer's own behavior" |
| fairness | *unmapped* | no SEC analog — left empty rather than proxied |
| effort | effort | "required a great deal of energy to deal with" |
| expectation | predict_event | "could have predicted the occurrence" |
| novelty | suddenness + inv(familiarity) | "sudden or abrupt" / "familiar" |
| urgency | urgency | "required an immediate response" |
| intensity | intensity | "how intense was the experience" |
| coping_potential | accept_conseq + self_control | "could live with the unavoidable consequences" |
| social_desirability | inv(social_norms) | "violated laws or socially accepted norms" (inverted) |
| moral_worth | inv(standards) | "clashed with her/his standards and ideals" (inverted) |
| attribution | self - max(other, chance) responsblt | locus of causation, signed |
| anticipated_emotion | author emotion sign x author intensity | author-grounded |

## POC-side prepared corpora (not committed; fetched + converted by scripts)

| Corpus | What | Access terms |
|---|---|---|
| **x-enVENT** (LREC 2022) | Experiencer-specific appraisal annotations (~20 dims) | CC-BY-4.0 |
| **Appraisal-enISEAR** (COLING 2020) | 1,001 enISEAR events, 7 Smith-Ellsworth dims, 3 annotators | citation required |
| **enISEAR / deISEAR** (ACL 2019) | 1,001 EN + 1,001 DE event descriptions, emotion gold | citation required |
| **HTK appraisal experiments** (WASSA 2021) | manual vs automatic appraisal annotation experiments | citation required |
| **ISEAR** (Scherer & Wallbott) | 7,666 situation reports, 37-country survey, appraisal questionnaire codes (CON/EXPC/PLEA/FAIR/CAUS/COPING/MORL...) | free for research via JULIELab CSV mirror of official data; cite Scherer & Wallbott |
| **EmoBank** (EACL 2017) | 10k sentences, VAD ratings, writer + reader perspectives incl. per-annotator ratings | citation required |
| **GoEmotions** (ACL 2020) | 54k Reddit comments, 27 emotion labels | Apache-2.0 |
| **SSEC** (Schuff et al. 2017) | SemEval stance/sentiment tweets + 8-emotion labels | citation required |
| **ATOMIC** (Sap et al. 2019) | 250k if-then commonsense relations incl. xReact/oReact emotional reactions | public release |
| **StoryCommonsense** (ACL 2018) | 280k character emotion/motivation annotations in stories | public release |
| **ESConv** (ACL 2021) | 1.3k emotional-support conversations, strategy labels | research use |
| **EPITOME / Empathy-Mental-Health** (EMNLP 2020) | 9k annotated support responses, 3 empathy mechanisms + rationales | research use |
| **EmpatheticDialogues** (ACL 2019) | 23k grounded dialogues, 32 emotion contexts | CC-BY-NC |
| **CREMA-D** (IEEE TAC 2014) | 7,442 acted clips x 3 presentation modes, crowd emotion votes | public release |
| **MACHIAVELLI** (Pan et al. 2023) | value-action gap benchmark: text games + harm/morality annotations + published agent results | MIT |

## Requires a signed agreement / registration (not fetched)

These are public for research but gated behind license requests or
registration forms - someone has to sign/request personally:

| Dataset | How to get it | Why we want it |
|---|---|---|
| **MSP-Podcast** | UT Dallas license request (free for research/education) | naturalistic speech, V/A/D - fixes "acted emotion" concern for H3 |
| **IEMOCAP** | USC SAIL license request | 12h dyadic speech, categorical + dimensional labels |
| **DAIC-WOZ / E-DAIC** | USC ICT data use agreement | real clinical distress (PHQ-8) - only corpus where "risk detection" means something |
| **DEAP** | Queen Mary / DEAP EULA | EEG/physio + V/A/D on music videos - cross-modal affect ground truth |
| **MAHNOB-HCI** | registration + agreement | multimodal affect (EEG + video + gaze) |
| **Aff-Wild2 / ABAW** | workshop registration | in-the-wild VA + expressions |
| **LIRIS-ACCEDE** | request form | movie clips, continuous VA - empathic-stimulus pool |
| **IAPS / NAPS / OASIS** | IAPS: license order; NAPS/OASIS: request forms | normed image stimuli w/ VA ratings |
| **WASSA empathy datasets** | credentials form at lt3.ugent.be/resources/wassa-2021-shared-task | Batson empathic-concern/personal-distress gold on essays |
| **RECCON** | request via project page | emotion-cause annotations in conversation |
| **GEMEP** | paid license via University of Geneva | enacted emotion portrayals (costs money) |
| **TalkLife (EPITOME)** | research@talklife.co | 235k peer-support interactions (non-commercial) |
| **ALOE** (Yang et al. 2024, arXiv:2405.00948) | HF dataset `Blablablab/ALOE`, gated=manual - request access on the dataset page | target + observer appraisals + 3,262 alignment labels = empathic accuracy as data; CC-BY-NC-SA |

## Adjacent models (eval-only baselines, not our data)

Models trained on appraisal/emotion tasks that can serve as external
baselines or validators. The first two are fetched to
`data/external_models/` by `fetch_corpora.py`; all are CC-BY-NC-SA or
similarly non-commercial - usable for evaluation, not for shipping.

| Model | What | Why it matters to us |
|---|---|---|
| `Blablablab/empathy-appraisal-span` | OpenPrompt+RoBERTa, 9 appraisal-label span classifier trained on ALOE | trained appraisal model baseline; macro-F1 0.56 shows task difficulty |
| `Blablablab/empathy-appraisal-alignment` | Siamese mpnet scoring whether two appraisals align | off-the-shelf empathic-accuracy scorer: inferred vs. self-reported appraisal |
| `Nikhil0097/wavlm-large-emotion-vad` | WavLM on MSP-Podcast, speaker-independent V/A/D regression | ready-made dimensional-emotion audio baseline for H3 |
| `BenRongey/deberta-v3-base-emobank-vad` etc. | text VAD regressors | the competing 3-dim emotion theory as sanity baseline |
| `ZebangCheng/Emotion-LLaMA` | multimodal emotion reasoning + explanation | closest "why not just what" artifact; free-text reasoning, no appraisal dims |

No HF model ships "situation -> 17-dim appraisal vector"; the de-facto
approach is frontier LLM + rubric prompt, which is what this benchmark
measures.

## Known gaps (no public dataset found — own annotation required)

- Human appraisal ratings on **our own** vignette families (the 500-item,
  κ > 0.6 preregistered requirement; annotation pool = seeded draw across
  all splits of the 560-item corpus). enVENT is adjacent gold, not a
  substitute: different text register, different situation families.
- `fairness` appraisals: ISEAR does carry FAIR/MORL questionnaire codes
  (raw, needs codebook mapping), but no SEC-style corpus rates fairness as a
  clean dimension - still an own-annotation item for gold.
- Multi-party / multi-experiencer appraisals beyond x-enVENT's scope.
- Naturalistic (non-acted) acoustic distress: MSP-Podcast / DAIC-WOZ need
  license agreements; see the gated table above.
- ISEAR IS fetched (JULIELab CSV of the official mdb, 7,666 rows, appraisal
  questionnaire codes + country). Country codes need the ISEAR codebook to
  decode; 16 distinct COUN values present in this export.
