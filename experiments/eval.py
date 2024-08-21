import csv

from tqdm import tqdm

from cleartext2.predict_weighted import Pipeline


def load_mturk():
    path = "../data/lex.mturk.txt"
    with open(path, encoding="latin-1") as f:
        reader = csv.reader(f, delimiter="\t")
        rows = list(reader)

    data = []
    for row in rows[1:]:
        sentence, word, *gts = row
        parts = sentence.lower().split(word)
        if len(parts) != 2:
            # TODO: this case should produce multiple examples
            continue
        before, after = parts
        data.append({"before": before, "word": word, "after": after, "gts": gts})

    return data


class EvalPipeline:
    def __init__(self):
        self._data = load_mturk()

    def eval_case(
        self,
        top_k: int,
        likelihood_weight: float,
        frequency_weight: float,
    ):
        pipeline = Pipeline(top_k=top_k, likelihood_weight=likelihood_weight, frequency_weight=frequency_weight)
        correct = 0
        total = 0
        n = 1
        with tqdm(total=len(self._data)) as pbar:
            for ex in self._data:
                pred, _ = pipeline.predict(ex["before"], ex["word"], ex["after"])
                correct += int(pred in ex["gts"])
                total += 1
                if total % 10 == 0:
                    pbar.set_postfix_str(f"{correct / (n * total):.2%}")
                pbar.update(1)
        return correct / total


def main():
    top_ks = [1, 3, 5, 10]
    likelihood_weights = [0, 0, 0, 0]
    frequency_weights = [1, 1, 1, 1]
    pipeline = EvalPipeline()
    for top_k, likelihood_wt, freq_wt in zip(top_ks, likelihood_weights, frequency_weights):
        acc = pipeline.eval_case(top_k=top_k, likelihood_weight=likelihood_wt, frequency_weight=freq_wt)
        print(f"top_k={top_k}, likelihood_weight={likelihood_wt}, frequency_weight={freq_wt}: {acc:.2%}")


if __name__ == "__main__":
    main()
