"""The 20 silent errors TrainCheck reproduced (OSDI '25, Table 3, p. 11), one entry each.

`paper` and `synopsis` are Table 3's; `ae` and `location` are the bug ID and failure
location in TrainCheck's AE guide (`docs/ae-eval-s5.1-silent-issue-detection.md`, which
lists 18: TF-29903 and TF-33455 are not in it). `in_clone` says where the reproduction is in
the TrainCheck clone at /Users/ak/Documents/tensrview/TrainCheck (7a62cbb), its history
included; the other reproductions are in OrderLab/TrainCheck-Evaluation-Workloads, which is
not on this machine and was not fetched. `how` is what was run here:

- "shipped": TrainCheck's script as it is in the clone;
- "swapped": TrainCheck's script with only the model and the data replaced;
- "recreated": written here from the synopsis (and the sources named in the script);
- "simulated": the mechanism or the effect put in by hand, because the bug's own trigger
  cannot run here;
- "not run".

`reach`, on the rows not run, is judged from the synopsis and TensorQuill's code, not tested.
`review` is the reading in "Research papers and SageMaker Debugger review.md", D7: up to
seven of the 20 within reach of TensorQuill's kinds (B1, B3, B6, B8, B9, B10, B16; B1 and B3
doubtful). `runs` names the configurations `run_all.py` runs, as (tag, extra arguments); each
runs once as `buggy` and once as `healthy`.
"""

from __future__ import annotations

NOT_IN_CLONE = "not in the clone or its history"

BUGS: list[dict] = [
    {
        "id": "B1", "paper": "PT-84911", "ae": "pytorch-forum-84911", "location": "User code",
        "synopsis": "Misconfiguration that only BatchNorm layers are enabled causing training to not progress.",
        "in_clone": "docs/assets/code/84911.py (the 5-minute tutorial's script)",
        "cpu": "As shipped, no: it needs torchvision, efficientnet_pytorch, pretrained EfficientNet-b0 weights and "
               "CIFAR-100 (downloads), none installed. With the model and data swapped, yes.",
        "how": "swapped", "script": "b01_pt84911.py", "review": "within reach, doubtful",
        "runs": [("", []), ("w5", ["--weights", "5"]), ("long", ["--batches", "100"]),
                 ("initloop", ["--init-at", "loop"])],
    },
    {
        "id": "B2", "paper": "PT-84803", "ae": "pytorch-84803", "location": "HW/Driver",
        "synopsis": "Incorrect I/O setup causing data loss during P2P communication between GPUs.",
        "in_clone": NOT_IN_CLONE,
        "cpu": "No: a fault of GPU peer-to-peer copies. TrainCheck's AE reuses the PT-104336 trace for it.",
        "how": "simulated", "script": "hw_p2p_sim.py", "review": "not counted",
        "runs": [("", [])],
        "needs": "two or more GPUs with the faulty P2P setup",
    },
    {
        "id": "B3", "paper": "DS-1801", "ae": "deepspeed-1801", "location": "Framework",
        "synopsis": "Logic bug in gradient clipping of BF16Optimizer causing model weights for LayerNorm layers to go out of sync.",
        "in_clone": "only example_pipelines/DS-1801/config.json in the history (deleted in 7de7430), which points at "
                    "an external Megatron-DeepSpeed checkout",
        "cpu": "No as written: it needs DeepSpeed (not installed), Megatron-DeepSpeed and tensor parallelism on GPUs. "
               "The mechanism runs on two CPU processes.",
        "how": "simulated", "script": "b03_ds1801_sim.py", "review": "within reach, doubtful",
        "runs": [("", []), ("every1", ["--replica-every", "1"])],
        "needs": "deepspeed with the BF16 optimizer of that time, Megatron-DeepSpeed, two or more GPUs",
    },
    {
        "id": "B4", "paper": "SO-60335387", "ae": "stackoverflow-60335387", "location": "User code",
        "synopsis": "Missed zero_grad() call causing gradients to accumulate and training to be unstable.",
        "in_clone": NOT_IN_CLONE,
        "cpu": "Yes, re-created on TrainCheck's mnist.py (synthetic data for MNIST).",
        "how": "recreated", "script": "b04_so60335387.py", "review": "not counted",
        "runs": [("", [])],
    },
    {
        "id": "B5", "paper": "TF-23723", "ae": "transformers-23723", "location": "Framework",
        "synopsis": "Logic bug in truncation of input causing model output to be longer than threshold.",
        "in_clone": NOT_IN_CLONE,
        "cpu": "Not run: needs the Transformers version with the bug (5.23.0 is installed) and the model and "
               "tokenizer files (a download).",
        "how": "not run", "script": None, "review": "not counted",
        "needs": "transformers at the issue's version, the model and tokenizer from the Hub",
        "reach": "TensorQuill records no input or output lengths, so no kind could see it.",
    },
    {
        "id": "B6", "paper": "SO-67180955", "ae": "stackoverflow-67180955", "location": "Framework",
        "synopsis": "Data loader workers having the same seed causing identical data fetched by workers.",
        "in_clone": NOT_IN_CLONE,
        "cpu": "Not as written: PyTorch 1.9 and later seed NumPy in each worker (TrainCheck's AE also leaves it out: "
               "'requires older Python version'). The old seeding put back with a worker_init_fn runs on CPU.",
        "how": "simulated", "script": "b06_so67180955_sim.py", "review": "within reach (duplicate_batches family)",
        "runs": [("", [])],
        "needs": "PyTorch before 1.9 (and a Python it supports)",
    },
    {
        "id": "B7", "paper": "LT-725", "ae": "lightning-thunder-725", "location": "Framework",
        "synopsis": "Incorrect function transform causing dispatcher to not respect autocast contexts and use wrong datatype for training.",
        "in_clone": NOT_IN_CLONE,
        "cpu": "Not run: needs lightning-thunder (not installed) at the issue's version.",
        "how": "not run", "script": None, "review": "not counted",
        "needs": "lightning-thunder at the issue's version; autocast to bf16 is normally run on a GPU",
        "reach": "TensorQuill records no data types of activations, so no kind looks for the wrong one.",
    },
    {
        "id": "B8", "paper": "PT-115607", "ae": "pytorch-115607", "location": "Compiler",
        "synopsis": "Missing guards causing jit to not recompile even when the code has changed, causing model to not learn.",
        "in_clone": "example_pipelines/115607_bug.py in the history (deleted in 471f57e)",
        "cpu": "Runs on CPU with PyTorch alone once save_config=False, a keyword 2.15 refuses, is dropped. The bug "
               "does not show on 2.15: the compiled loop's Adam step counter equals the eager one.",
        "how": "shipped", "script": "b08_pt115607.py", "review": "within reach",
        "runs": [("", ["--disable"]), ("nodisable", [])],
        "needs": "a PyTorch version with the bug (TrainCheck's evaluation used PyTorch 2.2.2, paper p. 10)",
    },
    {
        "id": "B9", "paper": "PT-104336", "ae": "pytorch-104336", "location": "Framework",
        "synopsis": "Module.to not preserve the gradient syncing hooks during serialization, causing DDP model weights to go out of sync.",
        "in_clone": NOT_IN_CLONE,
        "cpu": "Yes, re-created on two CPU processes; the trigger used (overwrite_module_params_on_conversion) is one "
               "that takes the same path on 2.15, not necessarily the issue's.",
        "how": "recreated", "script": "b09_pt104336.py", "review": "within reach",
        "runs": [("", [])],
    },
    {
        "id": "B10", "paper": "X-DDP", "ae": "x-jxmnop-ddp-out-of-sync", "location": "User code",
        "synopsis": "User incorrectly invoking forward on internal modules of DDP instead of DDP itself, causing DDP model weights to go out of sync.",
        "in_clone": NOT_IN_CLONE + " (described in docs/blog/posts/traincheck-launch.md)",
        "cpu": "Yes, re-created on two CPU processes.",
        "how": "recreated", "script": "b10_x_ddp.py", "review": "within reach",
        "runs": [("", []), ("every1", ["--replica-every", "1"])],
    },
    {
        "id": "B11", "paper": "TF-34204", "ae": "transformers-34204", "location": "Framework",
        "synopsis": "Input processor of Pixtral model only returns the first item when input data contains multiple items, causing significant amount of data not used during training.",
        "in_clone": NOT_IN_CLONE,
        "cpu": "Not run: needs the Pixtral processor files (a download) and the Transformers version with the bug.",
        "how": "not run", "script": None, "review": "not counted",
        "needs": "transformers at the issue's version, Pixtral's processor and model files; a GPU for the model",
        "reach": "TensorQuill's data hook hashes the sample indices the loader yields, which are all still there "
                 "when the processor drops items, and records no batch shapes.",
    },
    {
        "id": "B12", "paper": "TF-33844", "ae": "transformers-33844", "location": "Framework",
        "synopsis": "Flash attention implementations of models in transformers didn't set dropout to 0 during evaluation, causing unstable and non-reproducible evaluation results.",
        "in_clone": NOT_IN_CLONE,
        "cpu": "Not run: needs flash-attn, which needs a CUDA GPU, and the Transformers version with the bug.",
        "how": "not run", "script": None, "review": "not counted",
        "needs": "flash-attn on a CUDA GPU, transformers at the issue's version",
        "reach": "The fault is in evaluation, which TensorQuill does not record; its probe runs in eval mode but "
                 "through scaled_dot_product_attention, not the flash-attn path.",
    },
    {
        "id": "B13", "paper": "TF-17877", "ae": "transformers-17877", "location": "Framework",
        "synopsis": "Logic bug in distinguishing initialized/uninitialized parameters causing repeated initialization, leading to performance issues and unknown accuracy impact.",
        "in_clone": NOT_IN_CLONE,
        "cpu": "Not run: needs the Transformers version with the bug and a pretrained checkpoint (a download).",
        "how": "not run", "script": None, "review": "not counted",
        "needs": "transformers at the issue's version, a pretrained checkpoint from the Hub",
        "reach": "It happens while the model loads, before training and before TensorQuill starts.",
    },
    {
        "id": "B14", "paper": "TF-29903", "ae": "(not in the AE table)", "location": "Framework",
        "synopsis": "Certain LSTM weights are not saved due to logic bug in the checkpointing function.",
        "in_clone": NOT_IN_CLONE,
        "cpu": "Yes, re-created (a checkpoint that drops two LSTM tensors, then a resume).",
        "how": "recreated", "script": "b14_tf29903.py", "review": "not counted",
        "runs": [("", []), ("minimal", ["--checkpoint", "minimal"])],
    },
    {
        "id": "B15", "paper": "TF-33455", "ae": "(not in the AE table)", "location": "Framework",
        "synopsis": "Logic bug in calculation of total training steps when gradient accumulation is enabled cause training to stop earlier than expected.",
        "in_clone": NOT_IN_CLONE,
        "cpu": "Not run: needs the Transformers version with the bug (Trainer's step count).",
        "how": "not run", "script": None, "review": "not counted",
        "needs": "transformers at the issue's version",
        "reach": "No kind compares the steps run with the steps planned; TrainCheck misses it too.",
    },
    {
        "id": "B16", "paper": "DS-5794", "ae": "deepspeed-5794", "location": "Framework",
        "synopsis": "Logic bug in MoE layer implementation causing all2all to be invoked with conflicting arguments, leading to training hang.",
        "in_clone": NOT_IN_CLONE,
        "cpu": "Not run: needs DeepSpeed's MoE layer (DeepSpeed is not installed) on several ranks.",
        "how": "not run", "script": None, "review": "within reach",
        "needs": "deepspeed at the issue's version, several GPUs; for TensorQuill, either tensorquill-server with "
                 "endpoint= (job_stalled, rank_silent) or NCCL Flight Recorder dumps (collective_hang, collective_desync)",
        "reach": "A hang is seen by the server's watchdog or from NCCL Flight Recorder dumps, neither of which a "
                 "CPU run with gloo here produces.",
    },
    {
        "id": "B17", "paper": "PT-96600", "ae": "pytorch-96600", "location": "HW/Driver",
        "synopsis": "Incorrect I/O setup causing data loss during P2P communication between GPUs.",
        "in_clone": NOT_IN_CLONE,
        "cpu": "No: a fault of GPU peer-to-peer copies. TrainCheck's AE reuses the PT-104336 trace for it.",
        "how": "simulated", "script": "hw_p2p_sim.py", "review": "not counted",
        "runs": [],  # shares B2's run
        "needs": "two or more GPUs with the faulty P2P setup",
    },
    {
        "id": "B18", "paper": "BC-86", "ae": "baichuan2-86", "location": "HW/Driver",
        "synopsis": "Faulty driver causing data loss during P2P communication between GPUs.",
        "in_clone": NOT_IN_CLONE,
        "cpu": "No: a driver fault on GPUs. TrainCheck's AE reuses the PT-104336 trace for it.",
        "how": "simulated", "script": "hw_p2p_sim.py", "review": "not counted",
        "runs": [],  # shares B2's run
        "needs": "GPUs with the faulty driver",
    },
    {
        "id": "B19", "paper": "PT-51800", "ae": "pytorch-51800", "location": "Framework",
        "synopsis": "SpectralNorm layers require train-time forward to initialize weights; Running evaluation before any training leads to very misleading results.",
        "in_clone": NOT_IN_CLONE,
        "cpu": "Yes, re-created; torch.nn.utils.spectral_norm still behaves this way on 2.15.",
        "how": "recreated", "script": "b19_pt51800.py", "review": "not counted",
        "runs": [("", [])],
    },
    {
        "id": "B20", "paper": "MM-702", "ae": "mmpretrain-702", "location": "Framework",
        "synopsis": "Missed call to freeze_stage inside the init method of SwinTransformer causes loss computation to not respect all computations during forward.",
        "in_clone": NOT_IN_CLONE,
        "cpu": "Not run: needs mmpretrain with mmcv and mmengine (not installed) at the issue's version.",
        "how": "not run", "script": None, "review": "not counted",
        "needs": "mmpretrain, mmcv and mmengine at the issue's version; ImageNet-style data",
        "reach": "TensorQuill has no notion of which parameters are meant to be frozen; trained ones changing is "
                 "what it expects.",
    },
]

# B2, B17 and B18 share one simulated run.
SHARED_RUN = {"B17": "B2", "B18": "B2"}
