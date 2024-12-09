# GRCCA

The PyTorch implementation of ''[Graph Representation learning via Contrastive Clustering Assignments](https://ieeexplore.ieee.org/document/10243574)''.

## Dependencies

- Networkx 2.2
- numpy 1.19.2
- Pandas 1.1.5
- python 3.6.2
- pytorch 1.7.0 
- scikit-learn 0.24.2
- scipy 1.5.2

## Dataset

The datasets are stored at [Download](https://drive.google.com/drive/folders/1RorWFU32_r2mEb39HzwYiAabnqNasgjm?usp=sharing).

Before running, please create a new folder 'data' for the datasets. 

For link prediction, please additionally create a new folder 'process_dataset' and download the splited datasets.

## Usage

(1) Node classification: 
python main_{datasets}.py

(2) Link prediction: 
python {}_link.py


