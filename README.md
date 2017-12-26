# skip-thoughts
An implementation of Skip-Thought Vectors in PyTorch.

Instructions
------------
* Download book corpus or any other data-set and concatenate all sentences into one file and put it in `./data/` directory
* Modify the following line in `Train.ipynb` notebook accordingly:
`d = DataLoader("./data/dummy_corpus.txt")`
* Download the movie review dataset and put `rt-polarity.neg` and  `rt-polarity.pos` in the `./tasks/mr_data` directory.

Note: There is no early stopping. The `Train` notebook runs at the rate of 1 epoch / 2 days. Your model is saved when `./saved_models` when the average training loss in the last 20 iterations dips below the previous best.
