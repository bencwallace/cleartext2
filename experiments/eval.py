import csv

from tabulate import tabulate
from tqdm import tqdm

from cleartext2.predict_weighted import Pipeline


def load_mturk():
    path = "../data/lex.mturk.txt"
    with open(path, encoding="latin-1") as f:
        reader = csv.reader(f, delimiter="\t")
        rows = list(reader)

    data = []
    for row in rows[1:]:
        # Note that splitting as below produces 51 ground truth labels as opposed to 50 in the paper
        sentence, word, *labels = row[:-2]  # (resp. second) last column is empty in all (resp. most) rows

        before, after = sentence.lower().split(
            word, 1
        )  # TODO: cases where `word` appears multiple times in `sentence` can be used to produce multiple examples

        # sort labels by frequency
        # TODO: preserve frequencies for evaluating predicted probabilities
        labels = list(
            zip(*sorted([(lbl, labels.count(lbl)) for lbl in set(labels)], key=lambda kv: kv[1], reverse=True))
        )[0]

        data.append({"before": before, "word": word, "after": after, "labels": labels})

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

        changed_count = 0
        changed_freq = 0
        correct = 0
        correct_top_k = 0
        prec = None
        prec_top_k = None
        acc = None
        with tqdm(total=len(self._data)) as pbar:
            for i, ex in enumerate(self._data):
                pred, _ = pipeline.predict(ex["before"], ex["word"], ex["after"])

                total = i + 1
                changed_count += int(pred != ex["word"])
                correct += int(pred in ex["labels"])
                correct_top_k += int(pred in ex["labels"][:top_k])

                changed_freq = changed_count / total
                prec = correct / changed_count if changed_count > 0 else 0.0
                prec_top_k = correct_top_k / changed_count if changed_count > 0 else 0.0
                acc = correct / total

                if total % 10 == 0:
                    pbar.set_postfix_str(
                        f"prec={prec:.2%}, prec_top_k={prec_top_k:.2%}, acc={acc:.2%}, changed={changed_freq:.2%}"
                    )
                pbar.update(1)
        return {"prec": prec, "prec_top_k": prec_top_k, "acc": acc, "changed": changed_freq}


def main():
    top_ks = [1, 3, 5, 10]
    likelihood_weights = [0] * len(top_ks)
    frequency_weights = [1] * len(top_ks)

    pipeline = EvalPipeline()
    results = []
    # TODO: redundant to iterate over top_ks in this way
    for top_k, likelihood_wt, freq_wt in zip(top_ks, likelihood_weights, frequency_weights):
        metrics = pipeline.eval_case(top_k=top_k, likelihood_weight=likelihood_wt, frequency_weight=freq_wt)
        results.append({"top_k": top_k, **metrics})
    print(tabulate(results, headers="keys", floatfmt=".2%"))


if __name__ == "__main__":
    main()
