# Few Shot Classifer

## Table of Contents

- [Few Shot Classifer](#few-shot-classifer)
  - [Table of Contents](#table-of-contents)
  - [About ](#about-)
  - [Getting Started ](#getting-started-)
    - [Prerequisites](#prerequisites)
  - [Usage ](#usage-)

## About <a name = "about"></a>

A few shot classification model for cancer classification. It has been built by finetuning the mpnet-base-v2 model.

## Getting Started <a name = "getting_started"></a>

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

Set up a virtual environment and install the requirements.txt

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
There is an option to use poetry as well.
Simply run the below command and it will setup the environment for you

`poetry install`

## Usage <a name = "usage"></a>

The model will be found on huggingface soon, once I am free I will upload it there. 
If anyone else wants to train this model, keep in mind that it does need 16gb VRAM and atleast 16gb RAM for decent speeds
It took me around 3 hours to train the model, it was trained on: 

- CPU : Intel i7-12700k
- GPU: AMD Radeon 7900xtx 
- RAM: 32gb DDR4 


