# KGAN: Knowledge Graph Augmented Network Towards Multi-view Representation Learning for Aspect-based Sentiment Analysis
## From: https://github.com/WHU-ZQH/KGAN

### Env
* python 3.8.0
* model_weight/temp

### Tạo môi trường ảo
* python -m venv env
* env\Scripts\activate (cmd)
* source env/Scripts/activate (Git Bash)

### Cài đặt thư viện
* python -m pip install --upgrade pip
* pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html  (with CPU)
* pip install -r requirements.txt --no-deps
* python -m spacy download en_core_web_sm
* python -m spacy download en_core_web_lg

### Thoát khỏi môi trường ảo (khi không dùng nữa)
* deactivate

## Training options

- **ds_name**: the name of target dataset, ['14semeval_laptop','14semeval_rest','15semeval_rest','16semeval_rest','Twitter'], default='14semeval_rest'
- **bs**: batch size to use during training, [32, 64], default=64
- **learning_rate**: learning rate to use, [0.001, 0.0005], default=0.001
- **dropout_rate**: dropout rate for sentimental features, [0.1, 0.3, 0.5], default=0.05
- **n_epoch**: number of epoch to use, default=20
- **model**: the name of model, default='KGNN'
- **dim_w**: the dimension of word embeddings, default=300
- **dim_k**: the dimension of graph embeddings, [200,400],  default=200
- **is_test**:  train or test the model, [0, 1], default=1
- **is_bert**: GloVe-based or BERT-based, [0, 1], default=0

More training options can be found in "./main_total.py".

## Running

#### training based on GloVe: 

* python -m main_total -ds_name 14semeval_laptop -bs 32 -learning_rate 0.001 -dropout_rate 0.5 -n_epoch 20 -model KGNN -dim_w 300 -dim_k 400 -kge analogy  -gcn 0  -is_test 0 -is_bert 0
* python -m main_total -ds_name 14semeval_rest -bs 64 -learning_rate 0.001 -dropout_rate 0.5 -n_epoch 20 -model KGNN -dim_w 300 -dim_k 200 -kge distmult -gcn 0 -is_test 0 -is_bert 0

#### training based on BERT: 

* python -m main_total -ds_name 14semeval_laptop -bs 32 -learning_rate 0.00003 -n_epoch 20 -model KGNN -dim_w 768 -dim_k 400 -kge analogy -gcn 0  -is_test 0 -is_bert 1
* python -m main_total -ds_name 14semeval_rest -bs 64 -learning_rate 0.00003 -n_epoch 20 -model KGNN -dim_w 768 -dim_k 200 -kge distmult -gcn 0 -is_test 0 -is_bert 1

The detailed training scripts can be found in "./scripts".

## Evaluation

To have a quick look, we saved the best model weight trained on the evaluated datasets in the "./model_weight/best_model_weight". You can easily load them and test the performance. You can evaluate the model weight with:

- python -m main_total -ds_name 14semeval_laptop   -bs 32  -model KGNN -dim_w 300 -dim_k 400 -is_test 1 
- python -m main_total -ds_name 14semeval_rest   -bs 64  -model KGNN -dim_w 300 -dim_k 200 -is_test 1 ## Training options

- **ds_name**: the name of target dataset, ['14semeval_laptop','14semeval_rest','15semeval_rest','16semeval_rest','Twitter'], default='14semeval_rest'
- **bs**: batch size to use during training, [32, 64], default=64
- **learning_rate**: learning rate to use, [0.001, 0.0005], default=0.001
- **dropout_rate**: dropout rate for sentimental features, [0.1, 0.3, 0.5], default=0.05
- **n_epoch**: number of epoch to use, default=20
- **model**: the name of model, default='KGNN'
- **dim_w**: the dimension of word embeddings, default=300
- **dim_k**: the dimension of graph embeddings, [200,400],  default=200
- **is_test**:  train or test the model, [0, 1], default=1
- **is_bert**: GloVe-based or BERT-based, [0, 1], default=0

More training options can be found in "./main_total.py".

## Running

#### training based on Glo

* python -m main_total -ds_name 14semeval_laptop -bs 32 -learning_rate 0.001 -dropout_rate 0.5 -n_epoch 20 -model KGNN -dim_w 300 -dim_k 400 -kge analogy  -gcn 0  -is_test 0 -is_bert 0
* python -m main_total -ds_name 14semeval_rest -bs 64 -learning_rate 0.001 -dropout_rate 0.5 -n_epoch 20 -model KGNN -dim_w 300 -dim_k 200 -kge distmult -gcn 0 -is_test 0 -is_bert 0

#### training based on BERT: 

* python -m main_total -ds_name 14semeval_laptop -bs 32 -learning_rate 0.00003 -n_epoch 20 -model KGNN -dim_w 768 -dim_k 400 -kge analogy -gcn 0  -is_test 0 -is_bert 1
* python -m main_total -ds_name 14semeval_rest -bs 64 -learning_rate 0.00003 -n_epoch 20 -model KGNN -dim_w 768 -dim_k 200 -kge distmult -gcn 0 -is_test 0 -is_bert 1

The detailed training scripts can be found in "./scripts".

## Evaluation

To have a quick look, we saved the best model weight trained on the evaluated datasets in the "./model_weight/best_model_weight". You can easily load them and test the performance. You can evaluate the model weight with:

- python -m main_total -ds_name 14semeval_laptop   -bs 32  -model KGNN -dim_w 300 -dim_k 400 -is_test 1 
- python -m main_total -ds_name 14semeval_rest   -bs 64  -model KGNN -dim_w 300 -dim_k 200 -is_test 1 