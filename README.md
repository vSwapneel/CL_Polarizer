


# Contrastive Learning as Polarizer
For [Contrastive Learning as a Polarizer: Mitigating Gender Bias by Fair and
Biased Sentences] paper

## Our Proposed Model 

- [Polarizer-bert-base-uncased](https://huggingface.co/kyungmin011029/Polarizer-bert-base-uncased/tree/main)
- [Polarizer-bert-large-uncased](https://huggingface.co/kyungmin011029/Polarizer-bert-large-uncased/tree/main)
- [Polarizer-roberta-base](https://huggingface.co/kyungmin011029/Polarizer-base)
- [Polarizer-roberta-large](https://huggingface.co/kyungmin011029/Polarizer-roberta-large/tree/main)

## 1. Clone and download first_baseline.ipynb
using !git clone, clone our project and run in for_github_colab.ipynb (If you run in Colab) 

## 2. Pretraining 
Search where train.sh file is, change !chmod+x based on your dir. 

```python
!chmod +x /content/drive/MyDrive/For_github_test/CL_Polarizer/CL_traning/pretraining/train.sh
!/content/drive/MyDrive/For_github_test/CL_Polarizer/CL_traning/pretraining/train.sh
```
The Args are as follows:
* `--model_name`: further pretraining model, use 'bert-base-uncased' or 'bert-large-uncased' or 'roberta-base' or 'roberta-large' 
* `--train_path`: path of train data, search csv path, use path of the 'clean.csv' file.
* `--dev_path`: path of dev data, use wikitext dev data file path.
* `--total_steps`: steps you want to pretraining, calculate epoch with batch size info. (effective_batch_size X steps = data X epoch) 
* `--save_every`: use same num with total_steps
* '--learning_rate': find best lr..
* '--margin': for best score, you can use 0.9 (1.0).
* '--save_path_prefix': save dir path. Later you can use this path for benchmark inference, eval.

## 3. Evaluation
Search wehere 'CL_Polarizer' dir is. 
```python
cd /content/drive/MyDrive/For_github_test/CL_Polarizer
```

*if you need other requriements, !pip install according to yout error msg.*

This is for inference 
```python
!python -m benchmark.intrinsic.stereoset.predict --seed 26 \
--model BertForMaskedLM \
--model_name_or_path [your_training_model_dir]
```
Before run this code, find predict.py file and change json dump path (for inference json file save) 
In predict.py file, change this path every time you want to test model inference. 

```python
    with open(f"{args.persistent_dir}/stereoset/[filename_you_want].json", "w") as f:
        json.dump(results, f, indent=2)
```

Inference args are follows. 

* '--model': which type of model you want to test. use "BertForMaskedLM" or "RobertaForMaskedLM"
* '--model_name_or_path': path of your pretrained model dir.

Finally check score, find eval.py and change 'args.predictions.file' path with prediction file path. 
In eval.py, this code. 

```python
    args.predictions_file = f"{args.persistent_dir}/stereoset/[filename_you_want].json"
```

check gender, race score both. In eval.py, change this code and test twice (same inference file path) 
for domain in ["gender"] -> ["race"] 
```python
        for domain in ["gender"]:
            results["intrasentence"][domain] = self.evaluate(
                self.domain2example["intrasentence"][domain]
```

## Training and Evaluation Configuration Table

This table summarizes the valid configurations used for training and evaluation in the experiments.

| **ID** | **Training**                               | **Evaluation**          |
|--------|--------------------------------------------|-------------------------|
| 1      | `bert-base-uncased`                        | `BertForMaskedLM`       |
| 2      | `bert-large-uncased`                       | `BertForMaskedLM`       |
| 7      | `roberta-base`                             | `RobertaForMaskedLM`    |
| 8      | `roberta-large`                            | `RobertaForMaskedLM`    |
| 9      | `kyungmin011029/Polarizer-bert-base-uncased` | `BertForMaskedLM`       |
| 10     | `kyungmin011029/Polarizer-bert-large-uncased` | `BertForMaskedLM`       |
| 15     | `kyungmin011029/Polarizer-base`            | `RobertaForMaskedLM`    |
| 16     | `kyungmin011029/Polarizer-roberta-large`   | `RobertaForMaskedLM`    |

---

The IDs here correspond to the file names in results_# which means that result file is associated to the setup mentioned in the table




