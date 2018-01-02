# skip-thoughts
An implementation of Skip-Thought Vectors in PyTorch.

### Blog
Here's a [blog explaining the subtleties of Skip-Thoughts](http://sanyam5.github.io/my-thoughts-on-skip-thoughts/)
![blog](http://sanyam5.github.io/images/skip-thoughts/skip-rnn.png)
![blog](http://sanyam5.github.io/images/skip-thoughts/impute-rnn.png)

Instructions
------------

### Training
* Download [BookCorpus](http://yknzhu.wixsite.com/mbweb) or any other data-set and concatenate all sentences into one file and put it in `./data/` directory
* Modify the following line in `Train.ipynb` notebook accordingly:
`d = DataLoader("./data/dummy_corpus.txt")`
* There is no early stopping. 
* The `Train` notebook runs at the rate of 1 epoch / 2 days on an Nvidia 1080 Ti. 
* Your model is saved when `./saved_models` when the average training loss in the last 20 iterations dips below the previous best.

### Evaluation
Only implemented on classification tasks
* Download the [movie review dataset](http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz) and put `rt-polarity.neg` and  `rt-polarity.pos` in the `./tasks/mr_data` directory.
* You may also test on other classification tasks by downloading the datasets and providing their path and tasks type in `Evaluate.ipynb`

