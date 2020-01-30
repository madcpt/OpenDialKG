# OpenDialKG 

OpenDialKG is a dataset of conversations between two crowdsourcing agents engaging in a dialog about a given topic. Each dialog turn is paired with its corresponding “KG paths” that weave together the KG entities and relations that are mentioned in the dialog. More details can be found in the following paper:

Seungwhan Moon, Pararth Shah, Anuj Kumar, Rajen Subba. ["OpenDialKG: Explainable Conversational Reasoning with Attention-based Walks over Knowledge Graphs"](https://www.aclweb.org/anthology/P19-1081.pdf), ACL (2019).

## Usage

### Data Processing
Commands are integrated into make_data.sh, so just run:
```
bash ./make_data.sh
```

### KG Embedding Pre-training
```
bash ./train_KGE.sh
```
By default, KG Embeddings are stored in `./save/KGE/transe/`.

###  KG Walker training

```
To Be Done
```

## Reference

To cite this work please use:
```
@InProceedings{Moon2019opendialkg,
author = {Seungwhan Moon and Pararth Shah and Anuj Kumar and Rajen Subba},
title = {OpenDialKG: Explainable Conversational Reasoning with Attention-based Walks over Knowledge Graphs},
booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
month = {July},
year = {2019},
}
```

## License
OpenDialKG dataset is released under [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode), see [LICENSE](LICENSE) for details.
