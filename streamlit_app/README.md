# streamlit 说明

## streamlit 使用方法

1. 安装`streamlit`：`pip3 install streamlit`
2. 本地运行`app.py`：`streamlit run <path of app.py> [--server.port <port_num>]`，例如`streamlit run 2023hackathon/streamlit_app/app.py --server.port 8501`
    1. 如果在`VSCode`中运行，可以在`端口`中添加转发规则`<port_num>`，将服务器上的`<port_num>`端口转发到`localhost:<port_num>`端口，从而能够在本地浏览器中查看应用

## 功能说明

1. 模型可以选择`LGBM`、`XGBoost`、`CatBoost`三种，通过`.pkl`文件载入内存
    1. 目前仅有`LGBM`模型，其他模型待上传(通过`pickle.dump()`导出为`<model_name>.pkl`)
    2. 目前仅有`LGBM`模型支持`oneAPI`后端，其他模型待上传(通过`pickle.dump()`导出为`<model_name>_oneAPI.pkl`)
3. 支持两种推理后端：`Sklearn`与`Intel oneAPI daal4py`
4. 支持两种推理模式：单实例(`Single instance`)与批量推理(`Batched instance`)，推理完毕后可以将结果作为`CSV`下载
    1. 单实例推理由用户手动输入数据
    2. 批量推理时读取用户上传的`CSV`文件作为输入(默认不超过`200MB`)，模板文件可以在界面上下载
    3. [test_data.csv](./test_data.csv)中包含`100`条数据，可以使用它作为批量推理的输入。该数据使用[test_data_gen.py](./test_data_gen.py)生成
5. 代码已经利用<https://github.com/hzhuangqiu/2023hackathon>与<https://share.streamlit.io/>在线部署，可以访问<https://hzhuangqiu-2023hackathon-streamlit-appapp-d4sadw.streamlit.app/>进行测试
