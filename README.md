# Chinese to English translation

This package provide the function to translate a Chinese sentence into an English sentence.

## Requirements

- Python version >= 3.6 and <= 3.9

## Install

```
pip install -i https://test.pypi.org/simple/ cn2en --extra-index-url https://pypi.org/simple
```

## Usage

```
from cn2en.model import Model

model = Model()
print(model.translate('湯姆不在床上。'))
```
