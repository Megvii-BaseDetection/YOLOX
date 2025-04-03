# Contribute to pixeltable-yolox

We welcome community contributions to pixeltable-yolox. If there's a new feature or integration that you'd like to see
and you're motivated to make it happen, this guide will help you get started.

pixeltable-yolox uses the standard fork-and-pull contribution model: fork the repo, clone the fork, create a
branch for your changes, and then submit the changes via a pull request. This guide will walk you through how to do
this step-by-step. Here are some guidelines to keep in mind for your first contributions:

* Familiarize yourself with the documentation and codebase. Look through the
    [community issues](https://github.com/pixeltable/pixeltable-yolox/issues) and
    [discussions](https://github.com/pixeltable/pixeltable-yolox/discussions) to see if it's a problem or feature that's
    been discussed before.
* If you're not sure how to proceed or where something should go, or if you have any other questions, don't hesistate
    to open a conversation on the [discussions](https://github.com/pixeltable/pixeltable-yolox/discussions) page. We're
    here to help!

## What Needs to Be Done?

We (the pixeltable-yolox maintainers) often get questions like, "How can I help?" Usually the best answer is, "go fix
whatever problem is bothering you the most, or build whatever feature you'd most like to see." If it's a problem you're
already familiar with, you'll be well positioned to hit the ground running and motivated to see it through. Often it's
something valuable that wasn't even on our radar.

That being said, there are a few areas that could use immediate attention:

* Testing: The original YOLOX repo had very few tests. We believe strongly in thorough, rigorous test coverage. Any
    improvements to test coverage will have high value.
* ONNX/torchscript/ncnn deploy integration: These features were present in the original YOLOX, but haven't yet been
    ported to pixeltable-yolox. They're present in the codebase, but non-functional.
* Model evaluation: It's in theory part of pixeltable-yolox (via `yolox eval` on the commandline) but is untested.

## Setting up a Dev Environment

The remainder of this document guides you through setting up your dev environment and creating your first PR.

Before making a contribution, you'll first need to set up a Pixeltable development environment. It's assumed that you
already have standard developer tools such as `git` and `make` installed on your machine.

1. Set up your Python environment for Pixeltable

    * Install Miniconda:

        * <https://docs.anaconda.com/free/miniconda/index.html>

    * Create your conda environment:

        * `conda create --name yolox python=3.9`
        * For development, we use Python 3.9 (the minimum supported version) to ensure compatibility.

    * Activate the conda environment:

        * `conda activate yolox`

2. Install Pixeltable

    * Fork the `pixeltable-yolox` git repo:

        * <https://github.com/pixeltable/pixeltable-yolox>

    * Clone your fork locally:

        * `git clone https://github.com/my-username/pixeltable-yolox`

    * Install dependencies:

        * `cd pixeltable-yolox`
        * `pip install poetry==2.1.1`
        * `poetry install --with dev`

    * Verify that everything is working:

        * `pytest -v`

We recommend VSCode for development: <https://code.visualstudio.com/>

## Crafting a pull request

Once you've set up your dev environment, you're ready to start contributing PRs.

1. Create a branch for your PR

    * First make sure your `main` branch is up-to-date with the repo:

        * `git checkout main`
        * `git pull home main`

    * Create a branch:

        * `git checkout -b my-branch`

2. Write some code!

    * Don't worry about making small, incremental commits to your branch; they'll be squash-committed when it
        eventually gets merged to `main`.

3. Create a pull request

    * `git checkout my-branch`
    * `git push -u origin my-branch`
    * Now visit the Pixeltable repo on github; you'll see a banner with an option to create a PR. Click it.
    * Once the PR is created, you can continue working; to update the PR with any changes, simply do a
        `git push` from your branch.

4. Periodically sync your branch with `home/main` (you may need to do this occasionally if your branch becomes
    out of sync with other changes to `main`):

    * Update your local main:

        * `git checkout main`
        * `git pull home main`

    * Merge changes from `main` into your PR branch:

        * `git checkout my-branch`
        * `git merge main`

    * Resolve merge conflicts (if any):

        * If there's a merge conflict in `poetry.lock`, follow the steps below.
        * Resolve all other merge conflicts manually.
        * When all conflicts are resolved: `git commit` to complete the process.

    * To resolve a merge conflict in `poetry.lock`:

        * First resolve merge conflicts in `pyproject.toml` (if any).
        * `git checkout --theirs poetry.lock`
        * `poetry lock --no-update`
        * `git add poetry.lock`

5. Code review

    * Once you've submitted your PR, a pixeltable-yolox maintainer will review it.
    * Respond to any comments on your PR. If you need to make changes, follow the workflow in Steps 3-4 above.
    * Once your PR is approved, click the green "Squash and merge" button on your PR page to squash-commit it into
        `main`.
    * You can now safely delete your PR branch. To delete it in your local clone: `git branch -d my-branch`

6. Congratulations! You're now a pixeltable-yolox contributor.
