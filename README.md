This repo includes some quantization methods implemented with cxx and python, which base on mxnet.

There are six quantization methods as belows:

**quantization_int8:** according to google's quantization aware training(QAT) method. But also align quantization method with tensorrt, that we deploy to. paper link: https://arxiv.org/abs/1712.05877

**FQN:** low-bit QAT for detection tasks.  paper link: http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Fully_Quantized_Network_for_Object_Detection_CVPR_2019_paper.pdf

**PACT:**  paper link: https://arxiv.org/abs/1805.06085

**DoReFa:** paper link: https://arxiv.org/abs/1606.06160

**QIL:** this method is hard to train, we haven't trained the quantiztaion parameters successfully. paper link:https://arxiv.org/abs/1808.05779

**GDRQ:** the method is easy to implement, but it's quantization pipeline is complicated. hard to reproduct its' result. paper link: https://arxiv.org/abs/1908.01477

## part of quantizatin training result

### PACT result

**setting**

```shell
the netowrk of cifar10 is resnet20, imagenet is resnet18
```

|                   | paper-cifar10 | our cifar10 | paper-imagenet | our imagenet |
| :---------------- | :------------ | :---------- | :------------- | :----------- |
| fp32              | 0.916         | 0.920       | 0.702          | 0.697        |
| 4bits             | 0.913         | 0.918       | 0.692          | 0.694        |
| gap(fp32 - 4bits) | 0.003         | 0.002       | 0.01           | 0.003        |
| 3bits             | 0.911         | 0.917       | 0.681          | 0.674        |
| gap(fp32 - 3bits) | 0.005         | 0.003       | 0.021          | 0.023        |
| 2bits             | 0.897         | 0.897       | 0.644          | 0.603        |
| gap(fp32 - 2bits) | 0.019         | 0.023       | 0.058          | 0.094        |

**notice:** our result with 2 bits quantization on imagenet is lower than its' reported result. about 4% gap.

### results between QAT methods

```shell
1. the netowrk of cifar10 is resnet20, imagenet is resnet18;
2. if it's not explictly stated, the bits of wegiht and activation is in the same by default.
3. google quantize means quantization_int8, w-google-act-pact means quantization_int8 for weight and PACT for activation.
```

**cifar10**

|         | pact-paper | our pact | our google quantize | w-gdrq-act-pact | w-google-act-pact |
| :------ | :--------- | :------- | :------------------ | :-------------- | :---------------- |
| fp32    | 0.916      | 0.920    | 0.920               | 0.920           | 0.920             |
| 4bits   | 0.913      | 0.918    | 0.914               | 0.917           | 0.919             |
| w3,act4 |            | 0.918    | 0.911               | 0.914           | 0.917             |
| 3bits   | 0.911      | 0.917    | 0.891               | 0.918           | 0.916             |
| 2bits   | 0.897      | 0.897    | 0.67                | 0.903           | 0.893             |

**imagenet**

|         | pact-paper | our pact | our google quantize | our gdrq  | w-gdrq-act-pact | w-google-act-pact | w-gdrq-act-pact(ft) |
| :------ | :--------- | :------- | :------------------ | :-------- | :-------------- | :---------------- | :------------------ |
| fp32    | 0.702      | 0.697    | 0.697               | 0.697     |                 |                   |                     |
| 4bits   | 0.692      | 0.694    | 0.658               |           |                 |                   |                     |
| w3,act4 |            | 0.685    | 0.639               | 0.689(ft) | 0.691           | 0.669             |                     |
| 3bits   | 0.681      | 0.674    | 0.536               |           | 0.674           | 0.644             | 0.682               |
| 2bits   | 0.644      | 0.603    | 0.042               |           | 0.608           | 0.412             | 0.632               |

### some training tricks for quantization-aware-training
In our private deep model, the int8 training result can align with result in fp32 training so far. And the real int8 inference on tensorrt result is nearly the same as int8 training result.

In our practice, the easy way to train model in int8 is finetuning the fp32 model. `base_itn8_lr = base_fp32_lr/2`, `finetune_epoch=2`, `lr_scheduler` is setting to `SineScheduler`, which `lr = base_int8_lr * Sine(curr_iter/total_iters * pi)`

### how to compile the cxx op into mxnet
due to `quantization_int8` operator requires `maxall_except_dim` function which implemented in `mshadow` by us. so replace the  source `reduceto1d.h`  file with ours.

```shell
1. copy those files in `operator_cxx/contrib` into `mxnet_home/operator/contrib`
2. copy reduceto1d.h in `3rdparty/mshadow/mshadow/extension/reduceto1d.h` into `mxnet_home/3rdparty/mshadow/mshadow/extension/reduceto1d.h`
3. compile your mxnet
```


## attach quantize node
We attach quantize node by parsing symbol file of mxnet. It can generate graph with quantization node with your quantization setting. Detail implementation in `utils/graph_optimize.py`.

There is a simple example with `resnet18 network`. to run `python3 utils/graph_optimize.py`. the `source graph` and `attached quantized node graph` as below shows:

**source graph**
![source graph](https://github.com/XiaotaoChen/mx_quant_op/raw/master/sources/fp32_graph.jpg)

**attached quantize node graph**
![attached quantize node graph](https://github.com/XiaotaoChen/mx_quant_op/raw/master/sources/attached_quant_node_graph.jpg)