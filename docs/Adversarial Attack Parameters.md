# Adversarial attack parameters

| Dataset | Attack Method | Parameter configuration |
|--------:|:--------------|:------------------------|
| **MNIST** | **FGSM** | `epsilon = 32/255 ≈ 0.125` |
| **MNIST** | **PGD** | `epsilon = 32/255`, step size `alpha = 2/255`, `steps = 10`, random init = `True` |
| **MNIST** | **APGD** | `l_inf` norm, `epsilon = 32/255`, `steps = 10`, `restarts = 1`, `seed = 0`, loss = `cross-entropy`, `EOT iterations = 1`, step-size update factor `rho = 0.75` |
| **MNIST** | **VMI-FGSM** | `epsilon = 32/255`, step size `alpha = 2/255`, `steps = 10`, momentum decay = `1.0`, neighborhood samples `N = 5`, neighborhood upper bound `beta = 1.5` |
| **MNIST** | **C&W** | `l_2` norm, confidence = `0.0`, untargeted, learning rate = `0.05`, binary search steps = `10`, max iterations = `5`, initial constant = `0.01`, halving/doubling limits = `5`, batch size = `128` |
| **MNIST** | **DeepFool** | `steps = 50`, overshoot = `0.05` |
| **MNIST** | **One-Pixel** | `pixels = 1`, `steps = 10`, population size = `10`, inference batch size = `128` |
| **MNIST** | **Square Attack** | `l_inf` norm, `epsilon = 32/255`, `queries = 500`, `restarts = 1`, square control `p_init = 0.8`, margin loss, rescaling schedule = `True`, `seed = 0` |

| Dataset | Attack Method | Parameter configuration |
|--------:|:--------------|:------------------------|
| **OrganAMNIST** | **FGSM** | `epsilon = 8/255 ≈ 0.031` |
| **OrganAMNIST** | **PGD** | `epsilon = 8/255`, step size `alpha = 2/255`, `steps = 10`, random init = `True` |
| **OrganAMNIST** | **APGD** | `l_inf` norm, `epsilon = 8/255`, `steps = 10`, `restarts = 1`, `seed = 0`, loss = `cross-entropy`, `EOT iterations = 1`, `rho = 0.75` |
| **OrganAMNIST** | **VMI-FGSM** | `epsilon = 8/255`, step size `alpha = 2/255`, `steps = 10`, momentum decay = `1.0`, `N = 5`, `beta = 1.5` |
| **OrganAMNIST** | **C&W** | `l_2` norm, confidence = `0.0`, untargeted, learning rate = `0.05`, binary search steps = `8`, max iterations = `5`, initial constant = `0.01`, halving/doubling limits = `5`, batch size = `128` |
| **OrganAMNIST** | **DeepFool** | `steps = 50`, overshoot = `0.05` |
| **OrganAMNIST** | **One-Pixel** | `pixels = 1`, `steps = 10`, population size = `10`, inference batch size = `128` |
| **OrganAMNIST** | **Square Attack** | `l_inf` norm, `epsilon = 8/255`, `queries = 500`, `restarts = 1`, `p_init = 0.8`, margin loss, rescaling schedule = `True`, `seed = 0` |

| Dataset | Attack Method | Parameter configuration |
|--------:|:--------------|:------------------------|
| **CIFAR-10** | **FGSM** | `epsilon = 2/255 ≈ 0.008` |
| **CIFAR-10** | **PGD** | `epsilon = 2/255`, step size `alpha = 2/255`, `steps = 10`, random init = `True` |
| **CIFAR-10** | **APGD** | `l_inf` norm, `epsilon = 2/255`, `steps = 10`, `restarts = 1`, `seed = 0`, loss = `cross-entropy`, `EOT iterations = 1`, `rho = 0.75` |
| **CIFAR-10** | **VMI-FGSM** | `epsilon = 2/255`, step size `alpha = 2/255`, `steps = 10`, momentum decay = `1.0`, `N = 5`, `beta = 1.5` |
| **CIFAR-10** | **C&W** | `l_2` norm, confidence = `0.0`, untargeted, learning rate = `0.05`, binary search steps = `8`, max iterations = `5`, initial constant = `0.01`, halving/doubling limits = `5`, batch size = `128` |
| **CIFAR-10** | **DeepFool** | `steps = 50`, overshoot = `0.05` |
| **CIFAR-10** | **One-Pixel** | `pixels = 1`, `steps = 10`, population size = `10`, inference batch size = `128` |
| **CIFAR-10** | **Square Attack** | `l_inf` norm, `epsilon = 2/255`, `queries = 500`, `restarts = 1`, `p_init = 0.8`, margin loss, rescaling schedule = `True`, `seed = 0` |
