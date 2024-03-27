CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python /content/drive/MyDrive/학부연구생/SimCTG/pretraining/train.py\
    --model_name bert-base-uncased\
    --train_path /content/drive/MyDrive/학부연구생/SimCTG_code/clean.csv\
    --dev_path /content/drive/MyDrive/학부연구생/SimCTG/data/language_modelling/wikitext103/wikitext103_raw_v1_validation.txt\
    --seqlen 64\
    --number_of_gpu 1\
    --batch_size_per_gpu 16\
    --gradient_accumulation_steps 2\
    --effective_batch_size 32\
    --total_steps 350\
    --print_every 50\
    --save_every 350\
    --learning_rate 1e-4\
    --margin 1.0\
    --save_path_prefix /content/drive/MyDrive/학부연구생/bert_base_plus_lr14_seed74_st350_wm03
  

