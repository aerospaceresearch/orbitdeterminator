# Contributing to OrbitDeterminator

:+1::tada: First off, thanks for taking the time to contribute! :tada::+1:

**OrbitDeterminator** is a community-led project; that means that anyone is welcome to contribute to OrbitDeterminator and join the community! If you would like to make improvements to the package, add new features that are useful to you and others, or have found a bug that you know how to fix, please submit a pull request!

The following is a set of guidelines for contributing to OrbitDeterminator project, which are hosted in the [AerospaceResearch.net organization](https://github.com/aerospaceresearch/orbitdeterminator) on GitHub. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

#### Table Of Contents

[Code of Conduct](#code-of-conduct)

[I don't want to read this whole thing, I just have a question!!!](#i-dont-want-to-read-this-whole-thing-i-just-have-a-question)

[What should I know before I get started?](#what-should-i-know-before-i-get-started)
  * [Overview](#overview)
  * [Getting Started](#getting-started)

[How Can I Contribute?](#how-can-i-contribute)
  * [Reporting Bugs](#reporting-bugs)
  * [Suggesting Enhancements](#suggesting-enhancements)
  * [Your First Code Contribution](#your-first-code-contribution)
  * [Pull Requests](#pull-requests)

[Styleguides](#styleguides)
  * [Git Commit Messages](#git-commit-messages)
  * [CoffeeScript Styleguide](#coffeescript-styleguide)

## Code of Conduct

This project and everyone participating in it is governed by the [AerospaceResearch.net Code of Conduct](Code_of_Conduct.md). By participating, you are expected to uphold this code.

## I don't want to read this whole thing I just have a question!!!

> **Note:** Please don't file an issue to ask a question. You'll get faster results by using the resources below.

We have an official [mailing list](https://lists.shackspace.de/mailman/listinfo/constellation) where you can post your question for the community.

If chat is more your speed, you can join the us at:

* [Join the OrbitDeterminator Zulip Chat](https://aerospaceresearch.zulipchat.com/#narrow/stream/147024-OrbitDeterminator)
    * Zulip is fully open-source chat service. Although it is a chat service, sometimes it takes several hours for community members to respond &mdash; please be patient!
    * Join the orbitdeterminator stream of zulip chat to ask questions related to this project
    * There are many other streams available, check the stream list

## What should I know before I get started?

### Overview
The orbitdeterminator package provides tools to compute the orbit of a satellite from positional measurements. It supports both cartesian and spherical coordinates for the initial positional data, two filters for smoothing and removing errors from the initial data set and finally two methods for preliminary orbit determination. The package is labeled as an open source scientific package and can be helpful for projects concerning space orbit tracking.

Lots of university students build their own cubesat’s and set them into space orbit, lots of researchers start building their own ground station to track active satellite missions. For those particular space enthusiasts we suggest using and trying our package. Any feedback is more than welcome and we wish our work to inspire other’s to join us and add more helpful features.

Our future goals for the package is to add a 3d visual graph of the final computed satellite orbit, add more filters, methods and with the help of a tracking ground station to build a server system that computes orbital elements for many active satellite missions.

### Getting Started

To get started with orbitdeterminator, you must first setup the package on your system using [Installation Guide](README.md). Once you are done with setup, you can start by using orbitdeterminator for your observations or you can use sample observation and follow the tutorial [here](https://orbit-determinator.readthedocs.io/en/latest/examples.html).

For better understanding of the codebase and structure you can check how different modules of the orbitdeterminator package works by using [this guide](https://orbit-determinator.readthedocs.io/en/latest/modules.html). Once you get better understanding of the code, you can go ahead with [How Can I Contribute Guide](#how-can-i-contribute)

## How Can I Contribute?

### Reporting Bugs

This section guides you through submitting a bug report for OrbitDeterminator. Following these guidelines helps maintainers and the community understand your report :pencil:, reproduce the behavior :computer: :computer:, and find related reports :mag_right:.

Before creating bug reports, please check issue list as you might find out that you don't need to create one. When you are creating a bug report, please [include as many details as possible](#how-do-i-submit-a-good-bug-report).

> **Note:** If you find a **Closed** issue that seems like it is the same thing that you're experiencing, open a new issue and include a link to the original issue in the body of your new one.

#### How Do I Submit A (Good) Bug Report?

Bugs are tracked as [GitHub issues](https://guides.github.com/features/issues/). So, create an issue on this repository and provide the following information.

Explain the problem and include additional details to help maintainers reproduce the problem:

* **Use a clear and descriptive title** for the issue to identify the problem.
* **Describe the exact steps which reproduce the problem** in as many details as possible. For example, start by explaining how you started OrbitDeterminator, e.g. which command exactly you used in the terminal. Also, when listing steps, **don't just say what you did, but explain how you did it**.
* **Provide specific examples to demonstrate the steps**. Include links to files or GitHub projects, or copy/pasteable snippets, which you use in those examples. If you're providing snippets in the issue, use [Markdown code blocks](https://help.github.com/articles/markdown-basics/#multiple-lines).
* **Describe the behavior you observed after following the steps** and point out what exactly is the problem with that behavior.
* **Explain which behavior you expected to see instead and why.**
* **Include screenshots and animated GIFs** which show you following the described steps and clearly demonstrate the problem.
* **If the problem wasn't triggered by a specific action**, describe what you were doing before the problem happened.
* **Include details about your configuration and environment.**


### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion for OrbitDeterminator, including completely new features and minor improvements to existing functionality. Following these guidelines helps maintainers and the community understand your suggestion :pencil: and find related suggestions :mag_right:.

Enhancement suggestions are tracked as [GitHub issues](https://guides.github.com/features/issues/). You can create an issue on this repository and provide the following information:

* **Use a clear and descriptive title** for the issue to identify the suggestion.
* **Provide a step-by-step description of the suggested enhancement** in as many details as possible.
* **Provide specific examples to demonstrate the steps**. Include copy/pasteable snippets which you use in those examples, as [Markdown code blocks](https://help.github.com/articles/markdown-basics/#multiple-lines).
* **Describe the current behavior** and **explain which behavior you expected to see instead** and why.
* **Include screenshots and animated GIFs** which help you demonstrate the steps or point out the part of OrbitDeterminator which the suggestion is related to.
* **Explain why this enhancement would be useful** to most users.

### Your First Code Contribution

* Fork the [orbitdeterminator repo](https://github.com/aerospaceresearch/orbitdeterminator) and clone your fork of the repo into your project directory.
* Now follow the [installation guide](README.md) for your clone of the repo and install any necessary dependencies.
* Check for any available issue which you feel you can solve. Mention in the comments that you are working on that issue.
* Make necessary changes following the [styleguides](#coffeescript-styleguide)
* If you're stuck at any problem ask maintainers for help on [Zulip Chat](https://aerospaceresearch.zulipchat.com/#narrow/stream/147024-OrbitDeterminator)
* Once you have done all the changes, commit your changes following the [styleguide](#git-commit-messages) for commit message.

To send a pull request follow the guide [below](#pull-requests)

### Pull Requests

Please follow these steps to have your contribution considered by the maintainers:

1. Follow the [styleguides](#styleguides).
2. Title of Pull Request must be short and clearly indicate what is it related to.
3. Describe the changes in the code e.g. if you have fixed a bug or issue, mention that issue in the pull request and describe how you solved it and why was it failing before.
4. Demonstrate working code for the fixed issue or new enhancement/function using screenshots and animated GIFs.
5. Make sure all the test case passes before submitting the pull request by using `pytest`
6. After you submit your pull request, verify that all [status checks](https://help.github.com/articles/about-status-checks/) are passing <details><summary>What if the status checks are failing?</summary>If a status check is failing, and you believe that the failure is unrelated to your change, please leave a comment on the pull request explaining why you believe the failure is unrelated. A maintainer will re-run the status check for you. If we conclude that the failure was a false positive, then we will open an issue to track that problem with our status check suite.</details>

While the prerequisites above must be satisfied prior to having your pull request reviewed, the reviewer(s) may ask you to complete additional design work, tests, or other changes before your pull request can be ultimately accepted.

Members of the Contributors team are encouraged to review pull requests that have already been reviewed, and pull request contributors are encouraged to seek multiple reviews. Reviews from anyone not on the Contributors team are always appreciated and encouraged!

## Styleguides

### Git Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line

### CoffeeScript Styleguide

* Set parameter defaults without spaces around the equal sign
    * `clear = (count=1) ->` instead of `clear = (count = 1) ->`
* Use spaces around operators
    * `count + 1` instead of `count+1`
* Use spaces after commas (unless separated by newlines)
* Use parentheses if it improves code clarity.
* Prefer alphabetic keywords to symbolic keywords:
    * `a is b` instead of `a == b`
* Avoid spaces inside the curly-braces of hash literals:
    * `{a: 1, b: 2}` instead of `{ a: 1, b: 2 }`
* Include a single line of whitespace between methods.
* Capitalize initialisms and acronyms in names, except for the first word, which
  should be lower-case:
  * `getURI` instead of `getUri`
  * `uriToOpen` instead of `URIToOpen`
