CUDA_VISIBLE_DEVICES=0 run6 python task_4.py \
    --model_type rnn \
    --pretraining_type word2vec_not_frozen \
    --pretraining_model GoogleNews-vectors-negative300.bin.gz \
    --model_output_path adam_pretrained_word2vecc_not_frozen.pt \
    --tensorboard_dir word2vec_1e_5 \
    --lr 1e-5 && CUDA_VISIBLE_DEVICES=0 run6 python task_4.py \
    --model_type cnn \
    --pretraining_type word2vec_not_frozen_cnn \
    --pretraining_model GoogleNews-vectors-negative300.bin.gz \
    --model_output_path cnn_adam_pretrained_word2vecc_not_frozen.pt \
    --tensorboard_dir cnn_word2vec_1e_5 \
    --lr 1e-5 && CUDA_VISIBLE_DEVICES=0 run6 python task_4.py \
    --model_type lstm \
    --pretraining_type word2vec_not_frozen_lstm \
    --pretraining_model GoogleNews-vectors-negative300.bin.gz \
    --model_output_path lstm_adam_pretrained_word2vecc_not_frozen.pt \
    --tensorboard_dir lstm_word2vec_1e_5 \
    --lr 1e-5 

CUDA_VISIBLE_DEVICES=0 run6 python task_4.py \
    --model_type bilstm \
    --pretraining_type word2vec_not_frozen_bilstm \
    --pretraining_model GoogleNews-vectors-negative300.bin.gz \
    --model_output_path bilstm_adam_pretrained_word2vecc_not_frozen.pt \
    --tensorboard_dir bilstm_word2vec_1e_5 \
    --lr 1e-5 

CUDA_VISIBLE_DEVICES=0 run6 python task_4.py \
    --model_type rnn \
    --pretraining_type word2vec_not_frozen_dropout \
    --pretraining_model GoogleNews-vectors-negative300.bin.gz \
    --model_output_path adam_pretrained_word2vecc_not_frozen_dropout.pt \
    --tensorboard_dir word2vec_1e_5_dropout \
    --lr 1e-5 