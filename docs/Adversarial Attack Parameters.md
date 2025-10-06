# Adversarial Attack Parameters

## MNIST Dataset

| Attack Method | Parameter Configuration |
|:--------------|:------------------------|
| **FGSM** | `epsilon = 32/255 ≈ 0.125` |
| **PGD** | `epsilon = 32/255`<br>`alpha = 2/255` (step size)<br>`steps = 10`<br>`random_init = True` |
| **APGD** | `norm = l_inf`<br>`epsilon = 32/255`<br>`steps = 10`<br>`restarts = 1`<br>`seed = 0`<br>`loss = cross-entropy`<br>`eot_iterations = 1`<br>`rho = 0.75` (step-size update factor) |
| **VMI-FGSM** | `epsilon = 32/255`<br>`alpha = 2/255` (step size)<br>`steps = 10`<br>`momentum_decay = 1.0`<br>`N = 5` (neighborhood samples)<br>`beta = 1.5` (neighborhood upper bound) |
| **C&W** | `norm = l_2`<br>`confidence = 0.0`<br>`targeted = False`<br>`learning_rate = 0.05`<br>`binary_search_steps = 10`<br>`max_iterations = 5`<br>`initial_constant = 0.01`<br>`halving_doubling_limits = 5`<br>`batch_size = 128` |
| **DeepFool** | `steps = 50`<br>`overshoot = 0.05` |
| **One-Pixel** | `pixels = 1`<br>`steps = 10`<br>`population_size = 10`<br>`inference_batch_size = 128` |
| **Square Attack** | `norm = l_inf`<br>`epsilon = 32/255`<br>`queries = 500`<br>`restarts = 1`<br>`p_init = 0.8` (square control)<br>`loss = margin`<br>`rescaling_schedule = True`<br>`seed = 0` |

## OrganAMNIST Dataset

| Attack Method | Parameter Configuration |
|:--------------|:------------------------|
| **FGSM** | `epsilon = 8/255 ≈ 0.031` |
| **PGD** | `epsilon = 8/255`<br>`alpha = 2/255` (step size)<br>`steps = 10`<br>`random_init = True` |
| **APGD** | `norm = l_inf`<br>`epsilon = 8/255`<br>`steps = 10`<br>`restarts = 1`<br>`seed = 0`<br>`loss = cross-entropy`<br>`eot_iterations = 1`<br>`rho = 0.75` |
| **VMI-FGSM** | `epsilon = 8/255`<br>`alpha = 2/255` (step size)<br>`steps = 10`<br>`momentum_decay = 1.0`<br>`N = 5` (neighborhood samples)<br>`beta = 1.5` (neighborhood upper bound) |
| **C&W** | `norm = l_2`<br>`confidence = 0.0`<br>`targeted = False`<br>`learning_rate = 0.05`<br>`binary_search_steps = 8`<br>`max_iterations = 5`<br>`initial_constant = 0.01`<br>`halving_doubling_limits = 5`<br>`batch_size = 128` |
| **DeepFool** | `steps = 50`<br>`overshoot = 0.05` |
| **One-Pixel** | `pixels = 1`<br>`steps = 10`<br>`population_size = 10`<br>`inference_batch_size = 128` |
| **Square Attack** | `norm = l_inf`<br>`epsilon = 8/255`<br>`queries = 500`<br>`restarts = 1`<br>`p_init = 0.8`<br>`loss = margin`<br>`rescaling_schedule = True`<br>`seed = 0` |

## CIFAR-10 Dataset

| Attack Method | Parameter Configuration |
|:--------------|:------------------------|
| **FGSM** | `epsilon = 2/255 ≈ 0.008` |
| **PGD** | `epsilon = 2/255`<br>`alpha = 2/255` (step size)<br>`steps = 10`<br>`random_init = True` |
| **APGD** | `norm = l_inf`<br>`epsilon = 2/255`<br>`steps = 10`<br>`restarts = 1`<br>`seed = 0`<br>`loss = cross-entropy`<br>`eot_iterations = 1`<br>`rho = 0.75` |
| **VMI-FGSM** | `epsilon = 2/255`<br>`alpha = 2/255` (step size)<br>`steps = 10`<br>`momentum_decay = 1.0`<br>`N = 5` (neighborhood samples)<br>`beta = 1.5` (neighborhood upper bound) |
| **C&W** | `norm = l_2`<br>`confidence = 0.0`<br>`targeted = False`<br>`learning_rate = 0.05`<br>`binary_search_steps = 8`<br>`max_iterations = 5`<br>`initial_constant = 0.01`<br>`halving_doubling_limits = 5`<br>`batch_size = 128` |
| **DeepFool** | `steps = 50`<br>`overshoot = 0.05` |
| **One-Pixel** | `pixels = 1`<br>`steps = 10`<br>`population_size = 10`<br>`inference_batch_size = 128` |
| **Square Attack** | `norm = l_inf`<br>`epsilon = 2/255`<br>`queries = 500`<br>`restarts = 1`<br>`p_init = 0.8`<br>`loss = margin`<br>`rescaling_schedule = True`<br>`seed = 0` |
