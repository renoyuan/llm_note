

## 下载源码&模型文件

源代码：https://github.com/QwenLM/Qwen

模型文件 ：https://huggingface.co/Qwen/Qwen-1_8B-Chat



## 安装依赖

python -m pip install -r requirements.txt # 依赖文件在源码中 



如果您的设备支持 fp16 或 bf16 flash-attention可以来来提高你的运行效率以及降低显存占用。

```
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention && pip install .
```

flash-attention是可选的，项目无需安装即可正常运



## 加载模型

直接使用开源模型

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

model_path = "Qwen/Qwen-7B-Chat" # 这个是下载好的模型文件的路径
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-7B-Chat",
    device_map="auto",
    trust_remote_code=True
).eval()

# Specify hyperparameters for generation. But if you use transformers>=4.32.0, there is no need to do this.
# model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)

response, history = model.chat(tokenizer, "你好", history=None)
print(response)
# 你好！很高兴为你提供帮助。

# 多轮对话
response, history = model.chat(tokenizer, "给我讲一个年轻人奋斗创业最终取得成功的故事。", history=history)
print(response)
# 这是一个关于一个年轻人奋斗创业最终取得成功的故事。
```



## 为什么要微调

1 回答不准确

模型回答对应业务的精度不够需要我们提供对应业务的数据来支持。



2 格式不规范

希望模型直接输出规范的数据格式如json 方便下游系统直接使用 但是开源的模型这个不是很方便





## 制作训练数据

数据格式为json 

train.json 格式如下 

```json
[
  {
    "id": "identity_0", // 对话id 
    "conversations": [ //对话内容，可以是多轮的
      {
        "from": "user",   // 用户
        "value": "你好"  // 用户说的话,提问
      },
      {
        "from": "assistant", // 模型
        "value": "我是一个语言模型，我叫通义千问。" // 模型回答内容
      }
    ]
  }
]
```







## LoRA微调模型

官方给出了三种 微调方案分别是  全参数微调 LoRA Q-LoRA

LoRA  是一个资源适中的方案，它只会更新模型部分参数，训练的结果也是输出更新后的部分参数权重。

微调1.8B的模型 大概消耗 显存13G左右 在3090上。

我使用的也是LoRA  方案

### 下载微调需要的库

```
pip install peft deepspeed
```



### 开始微调



```shell
cd Qwen # 进到源码里面

# 执行微调 lora 单卡微调
bash finetune/finetune_lora_single_gpu.sh -m .. qwen_/qwen1_8B_chat -d ../qwen/train.json 

# 微调结束后 默认在 output_qwen 目录下保存微调的模型结果

```



## 加载微调后的模型

与全参数微调不同，LoRA和Q-LoRA的训练只需存储adapter部分的参数。假如你需要使用LoRA训练后的模型，你需要使用如下方法。假设你使用Qwen-7B训练模型，你可以用如下代码读取模型：

```python
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    path_to_adapter, # path to the output directory
    device_map="auto",
    trust_remote_code=True
).eval()

```





```
# 加载微调模型可以会遇到以下文件缺失的问题，这个时候从训练模型的文件内复制过来就可以了。
modeling_qwen
configuration_qwen
qwen_generation_utils
```



## 验证模型效果

```python

A,_=model.chat(tokenizer,prompt,history=None)
print(A)
# 最好是能编写也业务相关的测试脚本得到数据
```







## 模型合并

LoRA 微调只微调了部分权重加载的时候还是要先加载原始模型再加载微调后的权重，方便部署模型可以将二者合并（LoRA支持合并，Q-LoRA不支持）



```python
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    path_to_adapter, # path to the output directory
    device_map="auto",
    trust_remote_code=True
).eval()

merged_model = model.merge_and_unload()
# max_shard_size and safe serialization are not necessary. 
# They respectively work for sharding checkpoint and save the model to safetensors
merged_model.save_pretrained(new_model_directory, max_shard_size="2048MB", safe_serialization=True)
```

`new_model_directory`目录将包含合并后的模型参数与相关模型代码。请注意`*.cu`和`*.cpp`文件可能没被保存，请手动复制。另外，`merge_and_unload`仅保存模型，并未保存tokenizer，如有需要，请复制相关文件或使用以以下代码保存

```
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
    path_to_adapter, # path to the output directory
    trust_remote_code=True
)
tokenizer.save_pretrained(new_model_directory)
```

## 量化微调后模型

这一小节用于量化全参/LoRA微调后的模型。（注意：你不需要量化Q-LoRA模型因为它本身就是量化过的。） 如果你需要量化LoRA微调后的模型，请先根据上方说明去合并你的模型权重。

我们推荐使用[auto_gptq](https://github.com/PanQiWei/AutoGPTQ)去量化你的模型。

```
pip install auto-gptq optimum
```



注意: 当前AutoGPTQ有个bug，可以在该[issue](https://github.com/PanQiWei/AutoGPTQ/issues/370)查看。这里有个[修改PR](https://github.com/PanQiWei/AutoGPTQ/pull/495)，你可以使用该分支从代码进行安装。

首先，准备校准集。你可以重用微调你的数据，或者按照微调相同的方式准备其他数据。

第二步，运行以下命令：

```
python run_gptq.py \
    --model_name_or_path $YOUR_LORA_MODEL_PATH \
    --data_path $DATA \
    --out_path $OUTPUT_PATH \
    --bits 4 # 4 for int4; 8 for int8
```