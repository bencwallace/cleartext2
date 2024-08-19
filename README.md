# cleartext2

cleartext2 is a reboot of [ClearText](https://github.com/bencwallace/cleartext). The goal of both these
projects is to leverage modern deep learning techniques for NLP, in particular text simplification, in
order to help students of English improve their reading comprehension skills.

cleartext2 can be thought of as a "simplification thesauraus": For a word in a given context, it tries to
find a suitable "simple" replacement that makes sense in the same context. This allows students of English
to read a text without having to refer to an English dictionary (which can often make the task of reading
more difficult) or translation dictionaries, which take the student out of the language entirely.

**Details**

Unlike ClearText, which attempted to use sequence-to-sequence RNNs to perform syntactic simplification,
cleartext2 performs [lexical simplification](https://en.wikipedia.org/wiki/Lexical_simplification) by
taking advantage of advances in LLMs, in particular BERT-like encoder models, which are a good fit for
the task.

> [!IMPORTANT]
> cleartext2 is currently a simple prototype. There are a multitude of ways to approach this problem and there remain numerous experiments to run in order to identify the best approach to the task cleartext2 takes on.

## Set up

1. Set up the development environment using [Poetry](https://python-poetry.org/docs/#installation):

```
poetry install
```

2. Activate the development environment:

```
poetry shell
```

3. Launch the backend:

```
fastapi dev cleartext2/app.py
```

4. [Install the Chrome extension](chrome/README.md)

## Usage

Simply navigate to a webpage with some difficult text, select a word you don't understand and
click on the cleartext2 extension icon. A popup will appear with a simpler word that can be
used in the given context.

## Example

Taken from [Wikipedia: Quantum mechanics](https://en.wikipedia.org/wiki/Quantum_mechanics):

![](assets/example.png)
