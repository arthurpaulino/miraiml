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

After forking and cloning the repository, checkout to a new branch `issue-<#issue>`
(e.g.: `issue-42`).

### While coding

Please try to follow the [PEP8 guidelines][pep8] as closely as possible.

Remember to document everything you code. You can see the rendered version of the
documentation by calling `# make docs` (requires [sphinx][sphinx]).

Before commiting your changes, remember to increment the package version according
to the [Semantic Versioning][semver] specification. As a shortcut, the version can
be incremented by calling `make` with one of the following directives: `major`,
`minor` or `patch`.

Feel free to call `$ make help` for more information on *make* directives.

### After coding

After opening a [pull request][pulls], insert the link of the issue on your pull
request comment and Github will automatically mention that reference on the issue.

[issues]: https://github.com/arthurpaulino/miraiml/issues
[pep8]: https://www.python.org/dev/peps/pep-0008/
[sphinx]: https://pypi.org/project/Sphinx/
[semver]: https://semver.org/
[pulls]: https://github.com/arthurpaulino/miraiml/pulls
