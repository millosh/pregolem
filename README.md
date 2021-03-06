# About Pre-Golem-Alpha-0.0.0.4.0

## Pregolem

Pregolem is the set of multilingual and semiotic tools written in
Python (and maybe some other programming languages in the future) with
the focus on post-structural semantic analysis.

## The name

The name is based on the Stanisław Lem's story Golem XIV [1]. This is
obviously the ancestor of Golem XIV in the very early stages, so the
name corresponds with that fact. Numbers in the name are also the part
of the super-major versions, while you should look into the git for
the ongoing versions.

If written as one word -- "pregolem" -- it's also a Serbian regional
adjective with the meaning "extremely big".

If you are interested into the stories around Pregolem, check the file
[stories/golem.org](stories/golem.org).

# Purpose, tools and usage

If you need standard linguistic tools, Pregolem is not for you. This
documentation will include a set of useful links for standard
linguistic analysis in the future, both because the project needs and
because obvious need to help others in naviagation through the complex
and rapidly evolving computational tools which deal with language.

The segments of Pregolem could be used for various kinds semantic
analysis, but you are going to need good technical knowledge of the
tools, as well as in-depth knowledge of the semiotic
systems. Otherwise, Pregolem will be useless for you.

Putting things to work at the current state of technology (2021) is
not trivial and require various compromises. The first one being usage
of different Python installations for different purposes (thus, better
[Anaconda](https://www.anaconda.com/) than
[pyenv](https://github.com/pyenv/pyenv), but you could use any of them
if you have developed your own workaround).

Every particular program will have defined requirements. If you are
really into this, you will have environments set up properly. If you
are interested into a particular feature, you should check the
requirements for that feature.

Except this README file, documentation has been written in GNU Emacs
Org Mode format. If you are not using any Org Mode client, it should
be anyway relatively fine for you to read the text in a regular text
editor or viewer.

You should be proficient in any reasonably well developed GNU
environment. It could be about Linux shell, but also
[MSYS2](https://www.msys2.org/) (if you are using MS Windows), MacOS
terminal or even [Termux](https://termux.com/) on Android. It is
possible that more basic documentation about using these programs will
be written in the future, but as long as Internet exists in the
current form, it seems redundant (if you click on any of the links
above, you are going to find documentation for particular program).

- [Setting up Anaconda](docs/anaconda.org)
- [Installing initial modules](docs/python_modules.org)

# Files

- Documentation
  - [Setting up Anaconda](docs/anaconda.org)
  - [Installing initial modules](docs/python_modules.org)
- Stories
  - [Golem stories](stories/golem.org)
  - [Theory](stories/theory.org)
- Tests
  - [spacy_udpipe](test-spacy_udpipe.py)
  - [spacy_wordnet](test-spacy_wordnet.py)

# Contributors

- Milos Rancic, millosh@gmail.com
- Uroš Krčadinac, uros.krcadinac@gmail.com

If you are interested to participate, send an email to
millosh@gmail.com with your motives.

# License

Unless otherwise specified, the content is licensed under GNU AGPL v.3
license.

# Footnotes

[1] https://en.wikipedia.org/wiki/Golem_XIV