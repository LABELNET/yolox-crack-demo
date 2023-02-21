# yolox-crack-demo

识别图像上的裂纹



**工程说明**

```
.
├── README.md
├── YOLOX_outputs         # 文件夹：测试和训练缓存
├── asserts               # 示例图片
├── crack.ipynb           # 模型训练过程
├── crack_exp_yolox_s.py  # 模型配置
├── datasets              # 文件夹：数据集存放
├── demo.py               # 推理测试
├── eval.py               # 模型评估
├── models                # 文件夹：模型权重文件
│   ├── best_ckpt.pth
├── predict.py            # 推理运行及其后处理
├── tools                 # 数据集转换工具 
│   ├── __init__.py
│   ├── labelimg2coco.py
│   └── labelimg2voc.py
└── train.py              # 模型训练
```

## 示例图

**测试样图**

![](https://github.com/LABELNET/yolox-crack-demo/raw/main/asserts/demo1.png)

**工件样图**

![](https://github.com/LABELNET/yolox-crack-demo/raw/main/asserts/demo3.png)

**工件原图**

![](https://github.com/LABELNET/yolox-crack-demo/raw/main/asserts/demo2.png)

## 图片测试

安装 YoloX 环境后，可使用 CPU 进行测试，测试模型，请[下载模型](https://drive.google.com/file/d/1I4JDTDgiU_ZnSNxH8z9M6pRtC_4cA3Af/view?usp=sharing)，若要提高适应度，可根据 `crack.ipynb` 进行说明重新进行模型训练。


**方式一**

```
python demo.py image -f crack_exp_yolox_s.py -c models/best_ckpt.pth --path asserts/test.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device cpu
```

**方式二**

见 `predict.py` 文件，修改里面相关参数，进行推理运行即可，现有配置支持 `CPU` ，可咨询修改为 `GPU`

```
python predict.py
```

## 注意

YOLOX 安装 0.3.0 分支版本，不要使用仓库最新代码，可能存在未知无法解决的错误；

