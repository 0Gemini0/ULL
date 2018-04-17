***Practical 1 of ULL***

General run command:
```
./main.py [settings]
```
Or:
```
python3 main.py [settings]
```

***Prerequisites***

Package prerequisites:
```
pip install numpy
pip install scipy
pip install sklearn
pip install matplotlib
```

We expect both the embeddings and datasets to have their default download names to prevent an overload of cluttering settings:
```
bow2.words
bow5.words
deps.words
MEN_dataset_natural_form_full
SimLex-999.txt
nouns.txt
word-analogy.txt
```
The folders containing the embeddings/data can be specified with the following command line settings:
```
--data_path: path to folder containing the datasets for all tasks.
--emb_path: path to folder containing the embedding files.
--out_path: path to folder where output will be stored.
```
Please correctly specify these based on your file structure via command-line or by changing the defaults in ```settings.py```.

***Exercise 3***

Run:
```
./main.py --exercise=3 [settings]
```
Relevant settings:
```
--N: The number of top similarities to print for each dataset. default 10.
```
Running the code using the run command above will result in the pearson/spearman similarity scores to be printed to screen for both Datasets, for the three types of embeddings. The top 10 most similar pairs for each dataset as rated by the various embedding types will also be printed to the screen.

***Exercise 4***

Run:
```
./main.py --exercise=4 [settings]
```

***Exercise 5***

Run:
```
./main.py --exercise=5 [settings]
```
Relevant settings:
```
--clu_mode: which type of clustering algorithm to use [density, distance]
--red_mode: which type of dimensionality reduction to use [pca, tsne]
--k: use k clusters with k-means clustering.
--min_samples: minimum number of word embeddings per cluster for density based clustering.
--eps: maximum distance between points in a cluster for density based sampling.
--dim: number of dimension after reduction [2 3].
--tsne_dim: number of dimensions to pre-reduce the embeddings to with pca before tsne.
--tsne_num: maximum number of embeddings to cluster with tsne (for speed).
```
The default settings (using the runn command above) will yield the experiments as described in our report. We reduce the nouns to two dimension with T-SNE, pre-reduced to 50 dimensions with PCA, for visualization. Then we perform clustering with DBSCAN (density) to obtain meaningful clusters. The visualizations are shown with the clusters superimposed on them, but note that the clustering has been applied to the unreduced embeddings. After closing the visualizations the clusters for each embedding type will be stored in separate files.

***Additional Notes***

Run all exercises:
```
./main.py
```

See overview of all settings:
```
./main.py -h
```
