# streamlit 说明

## streamlit 使用方法

1. 安装`streamlit`：`pip3 install streamlit`
2. 本地运行`app.py`：`streamlit run <path of app.py> [--server.port <port_num>]`，例如`streamlit run 2023hackathon/streamlit_app/app.py --server.port 8501`
    1. 如果在`VSCode`中运行，可以在`端口`中添加转发规则`<port_num>`，将服务器上的`<port_num>`端口转发到`localhost:<port_num>`端口，从而能够在本地浏览器中查看应用

## 实现功能

1. 经过讨论，由于训练需要上传的数据量过大，因此目前仅支持推理(后续如果有时间也可以支持微调)
2. 模型可以选择`LGBM`、`XGBoost`、`CatBoost`三种，通过`<model_name>.pkl`载入内存
    1. 目前仅有`LGBM`模型，其他模型待上传
    2. 目前不支持`oneAPI`模型
3. 支持两种推理模式：单实例(`Single instance`)与批量推理(`Batched instance`)，推理完毕后可以将结果作为`CSV`下载
    1. 单实例推理由用户手动输入数据
    2. 批量推理时读取用户上传的`CSV`文件作为输入(默认不超过`200MB`)，模板文件可以在界面上下载
    3. [test_dataset.csv](./test_dataset.csv)中包含`100`条数据，可以使用它作为批量推理的输入
4. 目前仅支持本地运行，正在尝试利用<https://share.streamlit.io/>进行在线部署
