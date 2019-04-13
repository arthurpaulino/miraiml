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
them. Such issues will be tagged with the `reserved: approved` label.

Before commiting your changes, remember to increment the [version][version]
according to the [Semantic Versioning][semver] specification, with one difference:
there is also an UPDATE identifier, which MUST be incremented when the change does
not directly affect the way that the code works (eg.: updating the documentation).

After opening a [pull request][pulls], insert the link to the issue on your pull
request comment and then post the link to your pull request on the respective
issue.

[issues]: https://github.com/arthurpaulino/miraiml/issues
[version]: miraiml/__init__.py#L18
[semver]: https://semver.org/
[pulls]: https://github.com/arthurpaulino/miraiml/pulls
