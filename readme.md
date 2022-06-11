# HSE final qualifying project

Use several GNN models for benchmarking biomedical dataset.

Used GNN models:
* GCN
* GAT
* GCNII

Run training process in Google Colab

1. Install dependencies
```bash
!pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
!pip install -q wandb
!pip install igraph
```
1. Login to wandb
```bash
!wandb login
```

1. Run training with configuration via env vars and cli flags, e.g.
```bash
%env MODEL=gcn2
%env ROOT=/gdrive/MyDrive/GNN/KidneyMetabolic_deg
%env TYPE=kidney_metabolic 


!for i in $(seq 1 5); do python train.py --weighted --test_run -b 64 -e 200  -v 0.3 -m "$MODEL" -y 64 32 --conv_num 16  --test_ratio 0.2 --conv_pooling "mean" -r "$ROOT" -t "$TYPE"; done
!python eval.py --file "test_metric.json" 
```



