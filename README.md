# ChartQA Dual-Target

Qwen3-VL-2B fine-tuned with LoRA to answer chart questions with a JSON record: the chart
values it used, a box for each value, a plan that computes the answer from those values, and
the answer. The method and results are in the [report](report/report.pdf).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Using the trained model

`weights/stage2` is the final model and `weights/stage1` the Stage 1 model. Both are LoRA
adapters for `Qwen/Qwen3-VL-2B-Instruct`.

```python
from PIL import Image
from chartqa_dt.config import build_config
from chartqa_dt.eval.generate import generate_one
from chartqa_dt.model.loading import load_adapter, load_model

loaded = load_adapter(load_model(build_config("configs/base.yaml").model), "weights/stage2",
                      trainable=False)
loaded.model.eval()
chart = Image.open("chart.png").convert("RGB")
record, *_ = generate_one(loaded, "Which year had the highest revenue?", chart, mode="training")
print(record)
```

For `weights/stage1`, use `mode="grounding"`.

## Reproducing the results

The datasets are downloaded from Hugging Face at fixed revisions and are not stored here.
Their locations can be set with `CDT_DATA_ROOT`, `CDT_CACHE_ROOT` and `CDT_OUTPUT_ROOT`.

```bash
python scripts/download_data.py
python scripts/prepare_refchartqa.py
python scripts/generate_synthetic.py
python scripts/mine_plans.py prepare --run runs/mining-1
NVIDIA_API_KEY=<key> python scripts/mine_plans.py ask --run runs/mining-1
python scripts/mine_plans.py score --run runs/mining-1
python scripts/build_mixtures.py
python scripts/train.py --config configs/stage1.yaml
python scripts/train.py --config configs/stage2.yaml --init-adapter outputs/stage1/checkpoints/stage1-best-step6349
python scripts/evaluate.py chartqa --slice data/chartqa_val_200.json --tag stage2 --adapter weights/stage2 --adapter-stage stage2
python scripts/evaluate.py refchartqa --n 200 --tag stage2 --adapter weights/stage2 --adapter-stage stage2
```

## Layout

```
src/chartqa_dt/   data, synth (synthetic charts), plans, model, train, eval
scripts/          one script per pipeline step
configs/          Stage 1 and Stage 2 settings
data/             record ids of the evaluation sample and training mixtures
weights/          trained Stage 1 and Stage 2 adapters
third_party/      official ChartQA and RefChartQA evaluators, unmodified
report/           project report
```

## License

Code: Apache-2.0. The evaluators in `third_party/` keep their own licenses. ChartQA and
RefChartQA (GPL-3.0 / AGPL-3.0) are not included, and the weights, trained on them, are also
subject to their terms.
