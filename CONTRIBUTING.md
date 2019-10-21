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
- Pointing out errors or lack of information on the `documentation`
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

Install the development dependencies with `make develop` and now you're all set.

### While coding

Please follow the [PEP8 guidelines][pep8] while coding, although the maximum line
length is 100 instead of 79. Using [flake8][flake8] is highly recommended and you
can do so by calling `~/miraiml$ make flake`.

Remember to document everything you code. You can see the rendered version of the
documentation by calling `~/miraiml$ make docs` (requires [Sphinx][sphinx] and
[sphinx-rtd-theme][sphinx_rtd]).

After coding the functionality, please write the respective unit test in the
`tests` folder. In order to test it, you'll need to install the lib with your code
by calling `make install` and then call `make tests`.

Feel free to call `~/miraiml$ make help` for more information on *make* directives.

### After coding

Before commiting your changes, remember to increment the package version according
to the [Semantic Versioning][semver] specification. The version is defined as a
string on `miraiml/__init__.py`.

Commit your changes and push them to the corresponding branch (e.g.: `issue-42`)
of your own fork of the repository and then go to the
[original repository page][repo]. You should be able to see a button to open a
pull request.

Make sure you select the `dev` branch as the target of your merge request. Insert
the link of to the issue on your pull request description and Github will
automatically mention that reference on the issue after you open the pull request.
It will make things a lot easier to keep track of.

[issues]: https://github.com/arthurpaulino/miraiml/issues
[pep8]: https://www.python.org/dev/peps/pep-0008/
[flake8]: https://pypi.org/project/flake8/
[sphinx]: https://pypi.org/project/Sphinx/
[sphinx_rtd]: https://pypi.org/project/sphinx-rtd-theme/
[semver]: https://semver.org/
[repo]: https://github.com/arthurpaulino/miraiml
