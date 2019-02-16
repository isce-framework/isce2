# Contributing Guidelines

This document is inspired by similar instructions document in the GDAL and pygmt repositories. 

These are some of the many ways to contribute to the ISCE project:

* Submitting bug reports and feature requests
* Writing tutorials or jupyter-notebooks
* Fixing typos, code and improving documentation
* Writing code for everyone to use

If you get stuck at any point you can create an issue on GitHub (look for the *Issues*
tab in the repository) or contact us on the [user forum](http://earthdef.caltech.edu/projects/isce_forum/boards).

For more information on contributing to open source projects,
[GitHub's own guide](https://guides.github.com/activities/contributing-to-open-source/)
is a great starting point if you are new to version control.


## Ground Rules

We realize that we don't have a Continuous Integration (CI) system in place yet (maybe you could start by contributing this). So, please be patient if Pull Requests result in some detailed discussions.  

## Git workflows with ISCE

This is not a git tutorial or reference manual by any means. This just collects a few best practice for git usage for ISCE development. There are plenty of good resources on YouTube and online to help get started.

### Commit message

Indicate a component name, a short description and when relevant, a reference to a issue (with 'fixes #' if it actually fixes it)

```
COMPONENT_NAME: fix bla bla (fixes #1234)

Details here...
```

### Initiate your work repository


Fork isce-framework/isce from github UI, and then
```
git clone https://github.com/isce_framework/isce2
cd isce2
git remote add my_user_name https://github.com/my_user_name/isce2.git
```

### Updating your local master against upstream master

```
git checkout master
git fetch origin
# Be careful: this will loose all local changes you might have done now
git reset --hard origin/master
```

### Working with a feature branch

```
git checkout master
(potentially update your local master against upstream, as described above)
git checkout -b my_new_feature_branch

# do work. For example:
git add my_new_file
git add my_modifid_message
git rm old_file
git commit -a 

# you may need to resynchronize against master if you need some bugfix
# or new capability that has been added to master since you created your
# branch
git fetch origin
git rebase origin/master

# At end of your work, make sure history is reasonable by folding non
# significant commits into a consistent set
git rebase -i master (use 'fixup' for example to merge several commits together,
and 'reword' to modify commit messages)

# or alternatively, in case there is a big number of commits and marking
# all them as 'fixup' is tedious
git fetch origin
git rebase origin/master
git reset --soft origin/master
git commit -a -m "Put here the synthetic commit message"

# push your branch
git push my_user_name my_new_feature_branch
From GitHub UI, issue a pull request
```

If the pull request discussion results in changes,
commit locally and push. To get a reasonable history, you may need to
```
git rebase -i master
```
, in which case you will have to force-push your branch with 
```
git push -f my_user_name my_new_feature_branch
```

### Things you should NOT do

(For anyone with push rights to github.com/isce-framework/isce2) Never modify a commit or
the history of anything that has been
committed to https://github.com/isce-framework/isce2
