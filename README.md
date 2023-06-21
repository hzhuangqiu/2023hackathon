# 2023 Intel® oneAPI 校园黑客松竞赛

## Run hyper_tuning

数据预处理与模型调优

```shell
# Download dataset
wget -c https://filerepo.idzcn.com/hack2023/datasetab75fb3.zip
unzip datasetab75fb3.zip
# Install dependencies
pip3 install -r requirements.txt
# Run hyper_tuning
python3 hyper_tuning.py
```

## Run model_dump

模型生成与导出，并输出模型测试数据

```shell
# Run model_dump
python3 model_dump.py
```

## Run inference

```shell
# Make sure the environment variable DALROOT is set correctly.
# (e.g. run the script /opt/intel/oneapi/setvars.sh)
g++ inference.cpp \
    -O2 -std=c++17 \
    -I $DALROOT/include \
    -L $DALROOT/lib \
    -lonedal_core \
    -lonedal_thread \
    -Wno-deprecated-declarations \
    -o inference
./inference
```

Note: Before using C++ for inference, you need to run model_dump.py to generate the trained models (stored in `native_binary_modelname.txt`) and test data (stored in `test_dataset.csv`).

## stramlit

见[./streamlit_app/README.md](./streamlit_app/README.md)
