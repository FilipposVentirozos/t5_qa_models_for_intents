#!/bin/bash

N_EPOCHS=50
N_EPOCHS_AUX=10

#echo "1. Installing Sentence Transformers"

#git clone https://github.com/UKPLab/sentence-transformers.git
#cd sentence-transformers
#git checkout 0773bc71b52d27f2dcf21c35043ba76147796e67 # check me
#python -m venv venv
#source venv/bin/activate
#pip install -e .
#cd ..
#which python
#cp src/customised_scripts_bank/training_stsbenchmark_train_st5.py sentence-transformers/examples/training/sts/

#echo "2. Training Sentence Transformer"
#python sentence-transformers/examples/training/sts/training_stsbenchmark_train_st5.py --per_device_train_batch_size 36 --learning_rate 5e-4\
# --output_dir models/ --eval_steps 785 --num_train_epochs $N_EPOCHS_AUX
#deactivate

echo "3. Installing Transformers"
#git clone https://github.com/huggingface/transformers.git
cd transformers
#git checkout cbecf121cdeff6fb7193471cf759f9d734e37ea9
#python -m venv venv
source venv/bin/activate
#which pip
#pip install -e .
#pip install torch # This may differ according to your CUDA version, view https://pytorch.org/get-started/locally/
#pip install evaluate 
#pip install scikit-learn
#pip install tensorflow
#pip install nltk
#pip install rouge-score
##pip install git+https://github.com/google-research/bleurt.git

cd ..
#pwd
#cp src/customised_scripts_bank/trainer_qa.py transformers/examples/pytorch/question-answering/
#cp src/customised_scripts_bank/trainer_seq2seq_qa.py transformers/examples/pytorch/question-answering/
#cp src/customised_scripts_bank/run_seq2seq_qa.py transformers/examples/pytorch/question-answering/
#cp src/customised_scripts_bank/run_seq2seq_classifier.py transformers/examples/pytorch/question-answering/



echo "3b. WoT"
echo "python transformers/examples/pytorch/question-answering/run_seq2seq_classifier.py hyper-params/WoT/Classifier_t5_sampled.json"
python transformers/examples/pytorch/question-answering/run_seq2seq_classifier.py hyper-params/WoT/Classifier_t5_sampled.json $N_EPOCHS

echo "python transformers/examples/pytorch/question-answering/run_seq2seq_qa.py hyper-params/WoT/Boolean_t5_sampled.json"
python transformers/examples/pytorch/question-answering/run_seq2seq_qa.py hyper-params/WoT/Boolean_t5_sampled.json $N_EPOCHS

echo "python transformers/examples/pytorch/question-answering/run_seq2seq_qa.py hyper-params/WoT/BoolQ_Boolean_t5_sampled.json"
python transformers/examples/pytorch/question-answering/run_seq2seq_qa.py hyper-params/WoT/BoolQ_Boolean_t5_sampled.json $N_EPOCHS

echo "python transformers/examples/pytorch/question-answering/run_seq2seq_qa.py hyper-params/WoT/Extractive_t5_sampled.json"
python transformers/examples/pytorch/question-answering/run_seq2seq_qa.py hyper-params/WoT/Extractive_t5_sampled.json $N_EPOCHS

echo "python transformers/examples/pytorch/question-answering/run_seq2seq_qa.py hyper-params/WoT/SQuAD_Extractive_t5_sampled.json"
python transformers/examples/pytorch/question-answering/run_seq2seq_qa.py hyper-params/WoT/SQuAD_Extractive_t5_sampled.json $N_EPOCHS



echo "4b. WoT"
ehco "python transformers/examples/pytorch/question-answering/run_seq2seq_classifier.py hyper-params/WoT/Classifier_t5_sampled_pred.json"
python transformers/examples/pytorch/question-answering/run_seq2seq_classifier.py hyper-params/WoT/Classifier_t5_sampled_pred.json $N_EPOCHS

echo "python transformers/examples/pytorch/question-answering/run_seq2seq_qa.py hyper-params/WoT/Boolean_t5_sampled_pred.json"
python transformers/examples/pytorch/question-answering/run_seq2seq_qa.py hyper-params/WoT/Boolean_t5_sampled_pred.json $N_EPOCHS

echo "python transformers/examples/pytorch/question-answering/run_seq2seq_qa.py hyper-params/WoT/BoolQ_Boolean_t5_sampled_pred.json"
python transformers/examples/pytorch/question-answering/run_seq2seq_qa.py hyper-params/WoT/BoolQ_Boolean_t5_sampled_pred.json $N_EPOCHS

echo "python transformers/examples/pytorch/question-answering/run_seq2seq_qa.py hyper-params/WoT/Extractive_t5_sampled_pred.json"
python transformers/examples/pytorch/question-answering/run_seq2seq_qa.py hyper-params/WoT/Extractive_t5_sampled_pred.json $N_EPOCHS

echo "python transformers/examples/pytorch/question-answering/run_seq2seq_qa.py hyper-params/WoT/SQuAD_Extractive_t5_sampled_pred.json"
python transformers/examples/pytorch/question-answering/run_seq2seq_qa.py hyper-params/WoT/SQuAD_Extractive_t5_sampled_pred.json $N_EPOCHS


