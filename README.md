follow commands:
create conda environment
`conda env create -f environment.yml -n ai-math-tutor`

download data "full dataset"
https://github.com/google-research/google-research/tree/master/mathwriting

create data folder, move train, test, and valid folders into data folder

transform inkml files into images
`python convert_inkml_to_images.py`

create labels
`python create_labels.py`

test tokenizer and data loader if you want
`python tokenize_labels_and_test_dataloader.py`

train!
`python train.py`
there are parameters on the above that you can run. personally i do
`python train.py --batch_size 16 --lr 5e-4 --fp16` 

