# Third-party resources

This code does not bundle model weights or original narrative corpus text.
Dataset reconstruction is performed locally from public source rows. Downloaded
data/models retain their original terms and are not relicensed by this repository.

| Resource | Identification / source | Terms |
|---|---|---|
| WikiText-2 raw | Salesforce WikiText dataset card: https://huggingface.co/datasets/Salesforce/wikitext | Card lists CC BY-SA 3.0 and GFDL. Text originates from Wikipedia contributors; source row IDs and exact selected-text hashes are provided. |
| LAMBADA, OpenAI preprocessing | Official test blob: https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl ; dataset card: https://huggingface.co/datasets/EleutherAI/lambada_openai | Consult the dataset card's modified MIT terms and underlying narrative-text rights. No grant over those texts is made here. |
| Pythia-70M | https://huggingface.co/EleutherAI/pythia-70m | Consult the pinned revision's model card/license. |
| GPT-2 | https://huggingface.co/openai-community/gpt2 | Consult the pinned revision's model card/license. |
| Qwen2.5-0.5B | https://huggingface.co/Qwen/Qwen2.5-0.5B | Consult the pinned revision's model card/license. |

Reference arrays and score packets are experimental outputs. Public resource names
and citations identify third-party inputs, not the anonymous submission authors.

Dataset references:

- Stephen Merity, Caiming Xiong, James Bradbury, Richard Socher. Pointer Sentinel Mixture Models. https://arxiv.org/abs/1609.07843
- Denis Paperno et al. The LAMBADA dataset: Word prediction requiring a broad discourse context. ACL 2016. https://aclanthology.org/P16-1144/

Dependencies are installed separately and retain their own licenses. The code uses
standard NumPy, SciPy, PyTorch, Transformers and Matplotlib APIs.

