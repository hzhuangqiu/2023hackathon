# 2023 Intel® oneAPI 校园黑客松竞赛

## Resources

[resources.md](./resources/resources.md)

## 会议纪要

[LOG.md](./LOG.md)

## Run demo (Python)

```shell
# Download dataset
wget -c https://filerepo.idzcn.com/hack2023/datasetab75fb3.zip
unzip datasetab75fb3.zip
# Install dependencies
pip3 install -r requirements.txt
# Run demo
python3 demo.py
```

## Run inference demo (C++)

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

Note: Before using C++ for inference, you need to run demo.py to generate the trained models (stored in `native_binary_modelname.txt`) and test data (stored in `test_dataset.csv`).
