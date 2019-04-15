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

I've highlighted the labels that should be used in each case. Please, do not use
`reserved:*` labels.

## Contributing with code

Code contributions can only be made if there is an **open issue** related to the
changes you want to make, where we have discussed and come to an agreement about
them. Such issues will be tagged with the `reserved: approved` label. Comment on
the issue saying that you will implement it and I will assign it to you.

Before you start coding, checkout to a new branch called `issue-<#issue>` (e.g.:
`issue-42`).

Before commiting your changes, remember to increment the package version according
to the [Semantic Versioning][semver] specification, with one difference: there is
also an UPDATE identifier, which MUST be incremented when the change does not
directly affect the way that the code works (eg.: updating the documentation or
editing the `Makefile`).

The version can be incremented by calling `make` with one of the following
directives: `major`, `minor`, `patch` or `update`. Feel free to call `$ make help`
for more information.

After opening a [pull request][pulls], insert the link of the issue on your pull
request comment and Github will automatically mention that reference on the issue.

[issues]: https://github.com/arthurpaulino/miraiml/issues
[semver]: https://semver.org/
[pulls]: https://github.com/arthurpaulino/miraiml/pulls
