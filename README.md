# OBD

Official repository for the NeurIPS 2024 paper "[Offline Behavior Distillation](https://arxiv.org/pdf/2410.22728)" by Shiye Lei, Sen Zhang, and Dacheng Tao.



## Dependencies

- Python 3.7
- Pytorch 1.11
- mujoco 2.10
- d4rl
- wandb



## Quick Start

- Please refer to [command_parser.py](./configs/command_parser.py) for default hyper-parameters.
- near-expert policy $\pi^\ast$ checkpoints are provided in [offline_policy_checkpoints](./offline_policy_checkpoints) and obtained by using Cal-QL implemented in [CORL](https://github.com/tinkoff-ai/CORL).
- Av-PBC distilled datasets are available [here](https://drive.google.com/file/d/19yCQkCRy82YOqy8xZnOMqTDkgOhkAHWc/view?usp=sharing).

#### Syntheisze Behavioral Datasets

- **Av-PBC**

```shell
python obd_bptt.py --env 'halfcheetah-medium-replay-v2' --match_objective 'offline_policy' --q_weight --save_dir './saved_synset' --seed 0
```

- **PBC**

```shell
python obd_bptt.py --env 'halfcheetah-medium-replay-v2' --match_objective 'offline_policy' --save_dir './saved_synset' --seed 0
```

- **DBC**

```shell
python obd_bptt.py --env 'halfcheetah-medium-replay-v2' --match_objective 'offline_data' --save_dir './saved_synset' --seed 0
```

#### Evaluate Behavioral Datasets

- **Standard evaluation**

```shell
python evaluate_synset.py --env 'halfcheetah-medium-replay-v2' --match_objective 'offline_policy' --q_weight --eval_freq 1000 --save_dir './saved_synset' --group 'Evaluate' --seed 0
```

- **Ensemble evaluation**

```shell
python evaluate_synset.py --env 'halfcheetah-medium-replay-v2' --match_objective 'offline_policy' --q_weight --eval_freq 1000 --eval_ensemble --ensemble_policy_num 10 --save_dir './saved_synset' --group 'Ensemble-Evaluate' -- --seed 0
```

#### Cross Arch/Optim Evaluation

```shell
python evaluate_cross_arch.py --env 'halfcheetah-medium-replay-v2' --match_objective 'offline_policy' --q_weight --eval_freq 1000 --save_dir '/home/leaves/Data/OBD/q-value-weighted-synset' --group 'Cross-Arch-Optim-Evaluate' --seed 0
```



## Citation

```
@inproceedings{
lei2024offline,
title={Offline Behavior Distillation},
author={Lei, Shiye and Zhang, Sen and Tao, Dacheng},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024}
}
```



## Contact

For any issue, please kindly contact Shiye Lei: [leishiye@gmail.com](mailto:leishiye@gmail.com)



## Acknowledgment

- Remember The Past - Dataset Distillation: [https://github.com/princetonvisualai/RememberThePast-DatasetDistillation](https://github.com/princetonvisualai/RememberThePast-DatasetDistillation)
- Clean Offline Reinforcement Learning: [https://github.com/tinkoff-ai/CORL](https://github.com/tinkoff-ai/CORL)
