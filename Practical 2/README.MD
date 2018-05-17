***Practical 2 of ULL***
Python version:
```
python 3.5.0 or higher
```

Data preprocess run command:
```
./preprocess.py [settings]
```
Or:
```
python3 preprocess.py [settings]
```

Model training run command:
```
./main.py [settings]
```
Or:
```
python3 main.py [settings]
```

LST and AER predictions run command:
```
./lst.py [settings]
```
Or:
```
python3 lst.py [settings]

***Prerequisites***

Package prerequisites:
```
PyTorch 0.4.0 (please follow installation instructions on https://pytorch.org/)
pip install numpy
pip install msgpack_numpy
```

The folders containing the data can be specified with the following command line settings:
```
--data_path: path to folder containing the datasets for all tasks.
--out_path: path to folder where output will be stored.
```
Please correctly specify these based on your file structure via command-line or by changing the defaults in ```settings.py```.
