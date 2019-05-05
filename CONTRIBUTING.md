# Contributing

## Code of Conduct

1. Be polite, kind and respectful towards everyone
2. Welcome diversities and divergences of viewpoints
3. Do not go offtopic when commenting on issues
4. Be professional and do your best

## Contributing with issues

You can contribute with [issues][issues] by:

- Proposing/requesting a `new feature`
- Reporting a `bug`
- Asking for `help`
- Pointing out typos or errors on the `documentation`
- Asking a `question`
- Presenting a better way to do something with a `better code`
- Giving tips for better project `management` practices

I've highlighted the labels that will be used in each case.

## Contributing with code

### Before coding

Code contributions can only be made if there is an **open issue** related to the
changes you want to make, where we have discussed and come to an agreement about
them. Such issues will be tagged with the `approved` label. Comment on the issue
saying that you will work on it and I will assign it to you.

After forking the repository, clone the `dev` branch with:

```bash
~$ git clone --branch dev --single-branch https://github.com/your_username/miraiml.git
```

Then change directory and checkout to a new branch named after the issue you want
to work on (e.g.: `issue-42`)

```bash
~$ cd miraiml
~/miraiml$ git checkout -b issue-42
```

Now you can start coding.

### While coding

Please follow the [PEP8 guidelines][pep8] while coding, although the maximum line
length is 100 instead of 79. Using [flake8][flake8] is highly recommended:

```bash
~/miraiml$ flake8
```

Remember to document everything you code. You can see the rendered version of the
documentation by calling `~/miraiml$ make docs` (requires [Sphinx][sphinx] and
[sphinx-rtd-theme][sphinx_rtd]).

Before commiting your changes, remember to increment the package version according
to the [Semantic Versioning][semver] specification. As a shortcut, the version can
be incremented by calling `make` with one of the following directives: `major`,
`minor` or `patch`.

Feel free to call `~/miraiml$ make help` for more information on *make* directives.

### After coding

Commit your changes and push them to the `dev` branch of your own fork of the
repository and then go to the [original repository page][repo]. You should be able
to see a button to open a pull request.

Make sure you select the `dev` branch as the target of your merge request. Insert
the link of to the issue on your pull request description and Github will
automatically mention that reference on the issue. It will make things a lot easier
to keep track of.

[issues]: https://github.com/arthurpaulino/miraiml/issues
[pep8]: https://www.python.org/dev/peps/pep-0008/
[flake8]: https://pypi.org/project/flake8/
[sphinx]: https://pypi.org/project/Sphinx/
[sphinx_rtd]: https://pypi.org/project/sphinx-rtd-theme/
[semver]: https://semver.org/
[repo]: https://github.com/arthurpaulino/miraiml
