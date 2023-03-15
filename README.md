# [org-lduran2] / [git-clean-each-eps]
###### forked from [org-lduran2/git-clean-each-ipynb]
###### forked from [lduran2/cis4526-machine_learning_foundations]

configures the filters for cleaning Encapsulated
Postscript files.

## How to download

Add this module to your repo from its root directory by using
```bash
git submodule add https://github.com/lduran2/git-clean-each-eps filters/clean-each-eps
```

This command will download `git-clean-each-eps` to
`filters/clean-each-eps`, and update `.gitmodules` to associate the
submodule.

## How to install

### Requirements

#### In Windows

Installing requires use of the `git` command within a `bash` shell.
Allowing access to the `git` command, requires an extra step in
Windows operating systems. Namely, it requires adding `%git%/bin` to
the `%PATH%` variable where `%git%` represents the directory of the Git
installation.  This depends on the location of Git and the operating
system, but may appear as follows

```batch
PATH=C:\git\bin;%PATH%
```

where the assumed location of the Git installation is `C:\git`.

### Instructions

Then install it in the repo by using
```bash
bash filters/clean-each-eps/git-setup-filters
```

**Note:** This was changed from `git-`**`config`**`-filters`.

This script updates
`.gitattributes` to associate
`*.eps` files with the eponymous `eps` filter. 
It also associates
the `clean` `eps` filter
with the filter program `main` in the root
directory of this module.

### Only configure the filter

To configure the filter without changing
`.gitattributes`, use
```bash
bash filters/clean-each-eps/git-config-filters
```

It is not necessary to run this configuration script immediately after a
successful installation with no changes to the configuration.

[org-lduran2]: https://github.com/org-lduran2
[git-clean-each-eps]: https://github.com/lduran2/git-clean-each-eps
[org-lduran2/git-clean-each-ipynb]: https://github.com/org-lduran2/git-clean-each-ipynb
[lduran2/cis4526-machine_learning_foundations]: https://github.com/lduran2/cis4526-machine_learning_foundations
