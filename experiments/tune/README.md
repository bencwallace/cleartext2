# Fine-tuning

First, install and activate the fine-tuning environment:

```
poetry install
poetry shell
```

To fine-tune a model with the default settings:

```
python cleartext2_tune
```

The fine-tuning script is configured with [Hydra](https://hydra.cc/docs/intro/). As such, configuration settings can be changed either directly or through command-line parameters, as discussed in the Hydra docs. More information can be found as follows:

```
python cleartext2_tune --help
python cleartext2_tune --hydra-help
```

For example, if you don't have [W&B](https://wandb.ai/site), you can disable it as follows:

```
python cleartext2_tune wandb_mode=disabled
```

An example W&B report created from such a tuning run can be found [here](https://api.wandb.ai/links/bencwallace/8432fn63).

## Smoke test

To smoke test the tuning pipeline, run it with the [smoke config](conf/smoke.yaml) selected:

```
python cleartext2_tune -cn smoke
```
