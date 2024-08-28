# Fine-tuning

To fine-tune a model with the default settings:

```
poetry run python tune.py
```

The fine-tuning script is configured with [Hydra](https://hydra.cc/docs/intro/). As such, configuration settings can be changed either directly or through command-line parameters, as discussed in the Hydra docs. For more information, run the following commands:

```
poetry run python tune.py --help
poetry run python tune.py --hydra-help
```
## Smoke test

To smoke test the tuning pipeline, run it with the [smoke config](conf/smoke.yaml) selected:

```
poetry run python tune.py -cn smoke
```
