# Reranking close biomedical answer variants

Research checked: 19 October 2026. Primary papers, publisher records and author
repositories were checked. No models were downloaded or jobs submitted.

## Scope

Three problems should be distinguished in this project:

1. Different names for the same answer concept.
2. Similar expressions that change the entity, scope, specificity, units or bounds.
3. A scientifically valid expression that is absent from the scorer's accepted
   alias list.

Entity linking literature directly addresses the first two. It generally scores
concept IDs, so its results do not establish how to solve the third. Numbers and
descriptive answers may also fall outside an entity knowledge base.

## Most relevant papers

### 1. Improving Biomedical Entity Linking with Cross-Entity Interaction

Xu, Chen and Hu, AAAI 2023. **Prompt-BioEL**.
[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/26624) ·
[PDF](https://ojs.aaai.org/index.php/AAAI/article/download/26624/26396) ·
[Code](https://github.com/HITsz-TMG/Prompt-BioEL).

The reranker reads mention context and competing candidate entities together.
Candidates include synonym names. This permits direct comparison of near-matches.
Its fine-tuned reranker improves Acc@1 over a cross-encoder using the same retrieved
candidates by 0.6, 1.2 and 0.2 points on NCBI Disease, BC5CDR and COMETA. Its case
study includes confusion between the substance vomit and the condition vomiting;
it also documents an incorrect substitution of appetite for hunger.

**Adaptation, not a paper claim:** compare our candidate answers jointly with the
question and evidence. The published task supplies a mention to link; QA must infer
the answer from a question, and may have several acceptable answers.

### 2. Biomedical concept recognition with error-aware negative-enhanced ranking framework

Liu et al., Bioinformatics 2026. **ENR**.
[Publisher](https://doi.org/10.1093/bioinformatics/btag495) ·
[Publication record and abstract](https://pubmed.ncbi.nlm.nih.gov/42398028/) ·
[Code](https://github.com/sl-633/enr-recognizer).

ENR trains a listwise cross-encoder using ordinary negatives together with false
positive predictions from different recognizers. Its error analysis separates
granularity and scope errors from supported concepts missing from annotations.
The authors evaluate ontology concept recognition on MM-HPO and MM-GO, rather than
BioASQ answer ranking. Full publisher access was unreliable; the abstract,
publisher-indexed methods/results and author README were accessible.

**Adaptation:** use the real confusing outputs from our three generators as
training examples. Verify that a proposed semantic negative is actually wrong;
official nonacceptance alone cannot establish that.

### 3. Biomedical Entity Representations with Synonym Marginalization

Sung et al., ACL 2020. **BioSyn**.
[Paper](https://aclanthology.org/2020.acl-main.335/) ·
[Code](https://github.com/dmis-lab/BioSyn).

BioSyn gives positive probability mass to retrieved synonyms of the same concept
and refreshes candidates to expose harder negatives. This is entity normalization,
not a question-conditioned answer cross-encoder. It supports treating multiple
verified equivalent expressions as positives for a semantic objective.

### 4. Self-Alignment Pretraining for Biomedical Entity Representations

Liu et al., NAACL 2021. **SapBERT**.
[Paper](https://aclanthology.org/2021.naacl-main.334/) ·
[Code and pretrained models](https://github.com/cambridgeltl/sapbert).

SapBERT learns name representations using synonym concept identities and difficult
training pairs. It is an encoder pretraining method, not a complete answer
reranker. Candidate-to-candidate similarity could provide additional features;
similarity alone does not establish contextual equivalence.

### 5. Learning to Answer Biomedical Questions: OAQA at BioASQ 4B

Yang, Zhou and Nyberg, 2016.
[Paper](https://aclanthology.org/W16-3104/) ·
[PDF](https://aclanthology.org/W16-3104.pdf).

This directly studies BioASQ answer candidates and logistic collective reranking,
using answer relationships, semantic types and lexical features. Results are mixed:
with original features, factoid MRR changes from 0.3843 to 0.3791; with additional
features, it changes from 0.3689 to 0.3732. The strongest MRR configuration remains
original features without collective reranking. It is relevant precedent, not
evidence that collective scoring invariably improves QA.

### 6. Word embeddings and external resources for answer processing in biomedical factoid question answering

Dimitriadis and Tsoumakas, Journal of Biomedical Informatics 2019.
[Publisher](https://doi.org/10.1016/j.jbi.2019.103118) ·
[Abstract](https://pubmed.ncbi.nlm.nih.gov/30753948/).

Candidates are ranked with a binary classifier using text-derived, biomedical
resource and embedding features. This directly supports the general classifier
ranking approach used in our project. The abstract reports benefits from embedding
features; it does not establish a solution to our particular accepted-alias issue.

### 7. Accelerating Cross-Encoders in Biomedical Entity Linking

Sanz-Cruzado and Lever, BioNLP 2025.
[Paper](https://aclanthology.org/2025.bionlp-1.13/) ·
[Code](https://github.com/Glasgow-AI4BioMed/entitytools).

The authors jointly score multiple mention–entity pairs to reduce cross-encoder
cost. Across four datasets they report faster training and inference with accuracy
changes in both directions. This is an efficiency option if a contextual reranker
helps; it is not evidence of improved BioASQ MRR.

## Proposed experiment for our project

This section is a project proposal inferred from the literature, not a result
reported by the papers above.

Our current logistic ranker combines word/character TF-IDF with support,
frequency, lexical evidence and relation features. It does not encode the joint
meaning of question, evidence and candidate with a trained biomedical transformer.
The earlier feature-based pairwise ranker therefore does not settle whether a
contextual cross-encoder would help.

Start with a biomedical cross-encoder receiving **question + evidence + candidate**.
Train on the existing 1,296 fitting questions and select settings using the 144
validation questions. Reuse identical candidate pools, decoding and official
labels when comparing it with logistic ranking. Train on realistic confusions
from saved pools rather than predominantly easy unrelated negatives.

Maintain separate labels for official acceptance, semantic equivalence and evidence
support. A valid unmatched synonym is a semantic positive but remains officially
unaccepted under the unchanged scorer. An uncertain expression should not become
a semantic negative merely because it lacks an accepted string match. Judge-derived
labels need an explicit review and uncertainty policy.

If the contextual model helps on validation, compare a model that sees several
candidates together. For a concept-first variant, aggregate support across verified
synonyms while retaining every original string and its provenance. Then select
surface forms separately. Avoid irreversible merging or emitting invented aliases:
concept equivalence does not guarantee that the representative string is accepted.

Report official MRR@5, coverage@5 and semantic correctness separately, including
question-level gains/losses and a breakdown of synonym versus scope errors. Pure
reranking must leave full-pool coverage unchanged. Freeze choices before any new
test evaluation; the historical test95 has already been examined.

The central question is whether contextual candidate comparison improves selection
over the current logistic features, while handling incomplete aliases explicitly.
The cited entity-linking accuracies are not comparable to our BioASQ MRR values.
