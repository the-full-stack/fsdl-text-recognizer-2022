# Source Repo and Lab Repo

This repo is used to generate the more public-facing
[labs repo](https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2022-labs).

Changes made here will not be reflected there without intervention.
Currently, this is a manual process.

# Environment Setup

If you want to set up a development environment for this repo,
you can follow the same instructions as the students do:
[`instructions/setup/readme.md`](./instructions/setup/readme.md).
Any changes that are made to the user-facing process should be documented there, not here.

FYI, we are also experimenting with
[`devcontainer`s](https://code.visualstudio.com/docs/remote/containers),
which combine a container backend and a VSCode frontend,
as a solution for setting up environments.
GitHub can provide hosting for `devcontainer`s with
[Codespaces](https://docs.github.com/en/codespaces/setting-up-your-project-for-codespaces/introduction-to-dev-containers).
Core contributors can get access to GPU-accelerated Codespaces for development.
This environment is, as of the 2022 offering of the course,
not being directly maintained and is not documented in the notes below.

After following those instructions, run
`pre-commit install` to add pre-commit checks,
like linting.
These are also tested by CI,
but it's convenient to be able to find and fix small nits locally before push.

# Updating the Environment and Environment Management Principles

The notes below describe how to reliably update the environment
for the labs, from system libraries up to Python packages.

<details>
  <summary>
    <h2> What constrains our choices for the environment and build processes? </h2>

Before explaining the details of the environment,
it's helpful to take a step back and clearly state
what problem it is solving.

We want an environment for the labs that
- evinces best practices in and matches the reality of ML product development;
- can be setup quickly and easily by novice/intermediate developers; and
- stays reasonably in line with Google Colab.

Click anywhere on this text to reveal more details.
</summary>

### Matching ML Product Development

The purpose of the course is to teach ML product development, from soup to nuts.
One strength of the course is its closeness to "real" ML product development,
including the tools and workflows used.

Here are some of the features of ML development we want to mimic:
- Development is done by a team with varying levels of SWE expertise, so tools should be easy to learn and mainstream.
- Development includes best practices like testing, linting, CI/CD.
- Training requires GPU acceleration.
- Deployment is based on containerization.

### Quick and Easy Setup

We want to limit the difficulty of the setup,
while still keeping a process that is simple enough
that it can be easily explained to students and tinkered with.

That means running the entire class inside a user-managed container is out,
as are other means of providing a completely pre-built environment.

We compromise by using a transparent `Makefile`
that uses only limited `make` features.
The user experience roughly corresponds to joining a well-run team
with a canonical environment/build process already in place.

### Matching Google Colab

We want to keep our environment reasonably in line with Colab,
so that the labs run on that platform.

This serves two very important purposes:
- Colab provides an "out" in case the setup is not easy enough.
Setup on Colab is perforce automated.
- Colab provides GPU acceleration, which can be expensive, for free.

The Colab environment is a shifting target --
they seem to update PyTorch two weeks after release each time.
Due to the limited support for automation in Colab,
the best way to do things like check the current version of libraries
and run tests
is to manually execute a notebook.
[Here's one](https://fsdl.me/environment-testing-colab)
that checks that the environment is as expected and runs tests.
It should be run from beginning to end with Runtime > Run all,
but note that you have to provide a secret interactively in the final cell.
</details>

## OS

We aim for bug-free execution in the following environments:
- Ubuntu 18.04 LTS
- Google Colab
- (prod only) [Amazon Linux 2](https://hub.docker.com/layers/aws-lambda-python/amazon/aws-lambda-python/3.7/images/sha256-a329d5a1a30b7fb6adcaae2190a479d47395dac8e8cc31f10068954c62c14965?context=explore)
- (prod only) [Debian Buster](https://www.debian.org/releases/buster/)

As of writing, support for Windows Subsystem Linux 2 is in alpha.

## `conda`-managed Environment

`conda` provides virtual environments, system package installation (including Python runtimes),
and Python package installation.

We use it for virtual environments, system package installation, and Python runtime installation.

`poetry` also provides virtual environments and Python runtime installation,
but it does not work well for installing system packages,
and our core libraries are tightly intertwined with the system packages CUDA and CUDNN.
It may become a better choice than `conda` in the future.

<details>
  <summary> <h3> Python </h3>

We use <code>conda</code> to install and manage the Python runtime for users of the labs. Click to expand for more details.
  </summary>

Python runtimes for the production app and for CI are determined by Docker images,
but the `conda` environment is the source of truth.

So the Python version is mentioned in the following places:

- `environment.yml`, which describes the `conda` environment
- `.github/workflows/*.yml`, which describe the CI environment
- `api_server/Dockerfile` and `api_serverless/Dockerfile`, which describe the production app environment

Changes need to be synchronized by hand.
</details>

<details>
  <summary> <h3> CUDA/CUDNN </h3>

We use `conda` to install these GPU acceleration libraries.
They are needed for training but not for inference,
so the production app environment does not require them.
Click to expand for details.
  </summary>

The CUDA/CUDNN versions are mentioned in the following places:
- `environment.yml`, which describes the `conda` environment

Note that installing the NVIDIA drivers on which these depend is a fairly involved, often manual process.
We place it out of scope and presume they are present.

> If your (Linux) system does not have the required drivers,
which will be indicated by a warning when importing torch, see
[these instructions](https://askubuntu.com/questions/1077061/how-do-i-install-nvidia-and-cuda-drivers-into-ubuntu),
which were up-to-date as of 2022-04-13. Godspeed.
</details>

## Python Package Management with `pip`+`pip-tools`

Python packages are installed via `pip` with dependencies resolved and pinned by `pip-tools`.

Most high-level requirements are set in
`requirements/prod.in`
and `requirements/dev.in`.
Python build tools, e.g. `pip`, `pip-tools`, `setuptools`,
are specified elsewhere,
see below.

Python code quality (e.g. linting, doc checking)
is enforced via
[`pre-commit`](https://pre-commit.com/),
so the source of truth for versioning of those tools
is in `.pre-commit-config.yaml`.
They are also repeated in
`requirements/dev-lint.in`
so they can be optionally installed into the development environment.

The `.in` files are "compiled" by
[`pip-tools`](https://github.com/jazzband/pip-tools/)
to generate concrete `.txt` requirements files.
This ensures reproducible transitive dependency management
for Python packages installed by `pip`.

This choice
[effectively limits us to a single OS](https://github.com/jazzband/pip-tools/blob/37ce9e36d6033ede0667a1b293cd16843a85be4d/README.rst#should-i-commit-requirementsin-and-requirementstxt-to-source-control).
To support multiple platforms, we would need to produce "compiled" requirements files for each one and confirm tests pass in each case.
This can be automated by using cloud runners for each platform,
but we place this out of scope.

<details>
  <summary>
    <h3> Why not <code>conda</code>? </h3>
  Click to expand.</summary>

It is possible to use `conda` to install all packages,
which would have the salutary effect of limiting the number of tools
and unifying versioning and build information into one place.

However, that would create an extra, fairly heavy dependency in our Docker images.
We would either need to restrict the images we consider
(only those with `conda`; which might include lots of other things we don't want)
or include the `conda` build step in our Docker build.
Producing a `pip`-friendly file from `conda` requires
[`conda-lock`](https://pythonspeed.com/articles/conda-dependency-management/).
We end up with even greater differences between our dev and prod environment setup
and `conda-lock` is a less-established tool (it's in the `conda-incubator`).
It's also fairly heavy (e.g. depends on poetry) and moves many of our dependencies to the `conda-forge` channel.

`conda` also does not play nicely with Colab.

<h3> Is this approach crazy?</h3>

The [grok-ai nn template](https://github.com/grok-ai/nn-template)
has a similar approach.
They use `conda` for Python, CUDA, and CUDNN
and `pip` for almost everything else.
They install torch with `conda`,
which is worth considering for extra robustness,
but they don't target Colab or Docker.
</details>

<details>
  <summary>
    <h3> Python build tools. </h3> Click to expand.

To get reproducible builds, we need deterministic build tools.
</summary>

That means precisely pinned versions for:
- `pip`
- `setuptools`
- `piptools`

These versions are specified in
- the `Makefile`'s `>pip-tools` targets (for users)
- the `Dockerfile`s (for production)

They are not currently pinned in CI.
</details>

<details> <summary>
  <h3> <code>prod</code>uction. </h3> Click to expand.

These are the libraries required to run the app in production. </summary>

We aim to keep this environment lean,
to evince best practices for Dockerized web services.

They are specified at a high level in `requirements/prod.in`.

After updating the contents of `prod.in`,
run `make pip-tools` to perform any necessary updates to the compiled `prod.txt`
and update the local environment.

This may also change downstream environments, e.g. `dev`.
</details>

<details> <summary>
  <h3> <code>dev</code>elopment. </h3> Click to expand.

These are the libraries required to develop the model,
e.g. training.
It is also the curriculum development environment --
through the course, students learn to use the same tools
we use to manage the development of the material.
  </summary>

They are specified at a high level in `requirements/dev.in`,
which depends on `requirements/prod.in`

After updating the contents of either `prod.in` or `dev.in`,
run `make pip-tools` to perform any necessary updates to the compiled `dev.txt`
and update the local environment.
</details>

<details> <summary>
  <h3> <code>dev</code>elopment and <code>lint</code>ing. </h3> Click to expand.

  These are the libraries used to do code quality assurance,
  including linting.
  </summary>
This file is provided to allow these tools to be installed into
the development environment.
This eases integration of CQA with some developer tools.
The actual source of truth is in .pre-commit-config.yaml.
</details>

<details> <summary>
  <h3> Upgrade transitive dependencies without changing direct dependencies. </h3> Click to expand. </summary>

If the current compiled requirements file satisfies the constraints in the `.in` file,
then transitive dependencies will not be upgraded.

To force an upgrade, run `make pip-tools-upgrade`.
</details>
