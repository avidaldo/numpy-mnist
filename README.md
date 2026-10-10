# Images Are Arrays: MNIST with NumPy

A first project with NumPy, and with the tools of every software project: Git, a fork, a virtual environment, tests and continuous integration.

MNIST is a classic dataset of 70,000 handwritten digits, each a 28×28 grid of numbers. Loaded into Python, a set of images is just a NumPy array, of shape 60,000×28×28 for the training set, so everything you do to images (scaling them, adding noise, shifting and cropping them) is array manipulation. The [exercises](mnist_numpy.ipynb) go from shapes and data types to normalisation without leakage, loops against vectorisation, and data augmentation by hand. Tests check your answers, and a [solution](mnist_numpy_solution.ipynb) compares alternative ways of solving each exercise.

## What you need

- Python 3.13 or newer, Git, and an account on GitHub.
- NumPy's basics: creating arrays, indexing and slicing, reshaping. [NumPy: the absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html) covers them, and so do the notebooks of [Python data essentials](https://github.com/avidaldo/ai/tree/main/pia/python-data-essentials).

## Steps

### 1. Fork and clone

> [!TIP]
> Read about [forks, branches and pull requests](https://github.com/avidaldo/ai/blob/main/pia/setup/git_github.md) if you are not familiar with them.

A **fork** is your own copy of this repository on GitHub: you can push to it, and the original stays as it is. Press *Fork* at the top of this page, then clone your fork to your computer:

```bash
git clone https://github.com/<your-username>/numpy-mnist.git
cd numpy-mnist
```

> [!IMPORTANT]
> For feedback, the teacher needs to identify who is doing the exercises, so please write your GitHub profile URL [in the Moodle course](https://fpadistancia.edu.xunta.gal/mod/feedback/view.php?id=2318916&forceview=1).

### 2. Create the environment

Install the packages of `requirements.txt` in a virtual environment of their own. With [uv](https://docs.astral.sh/uv/):

```bash
uv venv
uv pip install -r requirements.txt
```

Or with conda:

```bash
conda create --name numpy-mnist python=3.13
conda activate numpy-mnist
pip install -r requirements.txt
```

Open `mnist_numpy.ipynb` in your editor and choose this environment as the notebook's kernel.

### 3. Run the tests: they fail

```bash
uv run pytest        # with conda, or with the environment activated: pytest
```

The tests run your notebook from top to bottom and check every name the exercises ask for. Before you solve anything, all of them fail but one, which checks that the notebook runs, and each failure names the exercise it is waiting for. The first run downloads MNIST (11 MB) into `data/`, which git ignores.

### 4. Turn on continuous integration

The file `.github/workflows/tests.yml` asks GitHub to run the same tests on its own servers every time you push: that is **continuous integration** (CI). GitHub keeps workflows switched off in a new fork until you allow them: open the *Actions* tab of your fork and enable them. From then on, every commit you push shows a green tick when the tests pass, or a red cross with the failing tests in its log.

### 5. Work on a branch, commit and push

Keep your work on a **branch**, so that `main` stays as the original:

```bash
git switch -c exercises
```

Solve the exercises in `mnist_numpy.ipynb`, in the empty cell below each one, and run the tests as you go. Commit each time you finish something, with a message that says what, and push whenever you want to save your progress on GitHub:

```bash
git add mnist_numpy.ipynb
git commit -m "Exercise 7: one-hot labels"
git push -u origin exercises     # the first time; then just git push
```

> [!TIP]
> Try a first pass on your own. You can use an AI assistant to help you, but **make sure you understand every solution**: the point is to learn how arrays and NumPy work and how to process images with them, and to work on a project with Git, GitHub, tests and continuous integration.

### 6. Open a pull request for feedback

A **pull request** proposes the commits of a branch for merging into another branch, and shows them as a diff that others can comment on. Once you have solved some exercises, open one from the `exercises` branch of your fork into the `main` branch of this repository: on your fork's page on GitHub, press *Contribute*, then *Open pull request*. Before you send it, read it as a reviewer would: the diff, the result of the tests, what you would change.

The tests run on your pull request, and the feedback comes as comments on its diff. Keep pushing to `exercises`: the pull request updates itself with every new commit. When the review is done, it is closed without merging, so that `main` keeps the exercises unsolved for everyone.

Pull requests that improve the exercises themselves (a typo, an unclear statement, a missing test) are welcome, and those are merged. Send them from a branch of their own, not from `exercises`.

### 7. Compare with the solution

Open [`mnist_numpy_solution.ipynb`](mnist_numpy_solution.ipynb) after trying each exercise. It compares several ways of solving most of them, and explains which one is better and why. To see the tests pass on it, on Linux or macOS:

```bash
MNIST_NOTEBOOK=mnist_numpy_solution.ipynb uv run pytest
```

and on Windows, in PowerShell:

```powershell
$env:MNIST_NOTEBOOK="mnist_numpy_solution.ipynb"; uv run pytest
```

## Working with an AI assistant

Once you have solved the exercises, ask an assistant for another way to solve two of them, time both versions, and write down which one is better and why.

## Keeping your fork up to date

If this repository changes after you fork it, GitHub's *Sync fork* button, on your fork's page, brings the changes into your `main`. Merge `main` into your branch afterwards if you need them there.

## Files

| File | What it is |
| --- | --- |
| [`mnist_numpy.ipynb`](mnist_numpy.ipynb) | The exercises |
| [`mnist_numpy_solution.ipynb`](mnist_numpy_solution.ipynb) | A solution, with alternatives compared |
| [`tests/test_mnist_numpy.py`](tests/test_mnist_numpy.py) | The tests, one or more per exercise |
| [`.github/workflows/tests.yml`](.github/workflows/tests.yml) | The continuous-integration workflow |
| `requirements.txt`, `pytest.ini` | The packages, and the tests' settings (warnings count as errors) |
