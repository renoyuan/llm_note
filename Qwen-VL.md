## 通义千问大规模视觉语言模型 

多模态大模型

### 版本 

https://github.com/QwenLM/Qwen-VL/blob/master/README_CN.md

Qwen/Qwen-VL 预训练版本

https://huggingface.co/Qwen/Qwen-VL

Qwen/Qwen-VL-Chat chat 版本

https://huggingface.co/Qwen/Qwen-VL-Chat

Qwen/Qwen-VL-Chat-Int4 量化版本

https://huggingface.co/Qwen/Qwen-VL-Chat-Int4/discussions



### 安装

第一次安装失败

报错 Floating point exception (core dumped) 

原因

其他 都按照官网要求

但 cudnn 版本没有对应上 重新安装cudnn 后成功。



### 运行&环境

3090 

Qwen-VL-Cha 版本 显存占用 21630MiB 

速率 10-20s /页



选了21个样例测试要素抽取的任务，优化了一下提示词发现开源模型效果有明显提升，但是同样提示词对官网的效果好像不明显。总结了一些现象。

1 提示词对引导大模型效果很关键，但是同一个模型不同版本，提示词需求可能都不一样

2 开源的模型存在一些虚假的回答，可能会把图片上不存在的信息回答出来，这个他们官网上的版本好像没看到

3 这个是多模态大模型，没有ocr结果提升的话只能提升模型效果，没有ocr结果加后处理