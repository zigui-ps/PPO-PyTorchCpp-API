# PPO for PyTorch c++

## How to build

1. Install Pytorch c++ API
2. mkdir build && cd build && cmake -DCMAKE_PREFIX_PATH='/path/to/pytorch_c++_api/libtorch' ..
3. make (or make -j)

## How to demo

```./A --env (name of gym env) --load_model (model path)```

- Hopper-v2

```./A --env Hopper-v2 --load_model trained_model/Hopper.xml```

## How to train

```./A --env (name of gym env) --train_step (# of train step) --load_model (model path if exists)```

- Hopper-v2

```./A --env Hopper-v2 --train_step 1000```
