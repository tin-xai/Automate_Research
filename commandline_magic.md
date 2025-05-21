# Command Line Magic 

## ⚗️ Git

### Git Credentials

#### Caching git password

##### Linux 
Set git to use the credential memory cache
```
git config --global credential.helper cache
```

To change the default password cache timeout
```
git config --global credential.helper 'cache --timeout=10800'
```

##### Mac OS
1. Check if `osxkeychain helper` is installed
```
git credential-osxkeychain
```
2. Set git to use the `osxkeychain helper`
```
git config --global credential.helper osxkeychain
```

#### Changing use credentials in a particular repository
First, clear local git credentials
```
git config --local credential.helper ""
```
then proceed with the git operations you want to perform. You will be prompted to enter credentials again.

### Revert git operations
#### Revert Git add
Unstage a particular file
```
git reset <file>
```
Unstage all changes
```
git reset
```
#### Revert unpushed Git commits
Delete the most recent commit, keeping the work you've done:
```
git reset --soft HEAD~1
```

Delete the most recent commit, destroying the work you've done:
```
git reset --hard HEAD~1
```

### Find remote URL of a Git repository
```
git config --get remote.origin.url
```

### Git branch operations
#### Push a local Git branch to remote
```
git checkout -b <branch>

git push -u origin <branch>
```

#### Delete a local Git branch

Delete the local branch only if you have already pushed and merged it with your remote branches.
```
git branch -d <branch_name>

git branch --delete <branch_name>
```

Delete the local branch regardless of its push and merge status.
```
git branch -D <branch_name>

git branch --delete --force <branch_name>
```

Push the local branch to remote and delete it locally.
```
git push -d
```

#### Delete a remote Git branch
```
git push <remote_name> --delete <branch_name>

git push <remote_name> :<branch_name>
```

#### Checkout a particular commit and create a branch
```
git checkout -b <branch_name> <commit_no>
```

### Git submodule operations
#### Clone a repository that contains submodules
Clone a respository including its submodules
```
git clone --recursive <url_to_git_repo>
```
Load submodules in a already cloned repository
```
git submodule update --init
# if there are nested submodules
git submodule update --init --recursive
```
Download N submodules at once
```
git submodule update --init --recursive --jobs 8
git submodule update --init --recursive -j 8
git clone --recursive --jobs 8 <url_to_git_repo>
```
#### Pulling with submodules
Pull all changes in the repo including changes in the submodules
```
git pull --recurse-submodules
```
Pill all changes for the submodules
```
git submodule update --remote
```

## ⚗️ Network
### Get process ID listening to a port
```
lsof -i :6006
```

## ⚗️ Find
### Find files modified after a particular timestamp
```
touch -t YYMMDDhhmm.SS /tmp/timestamp 

find directory -type f -newer /tmp/timestamp

rm /tmp/timestamp
```

## ⚗️ Kubernetes
### Find pods by criteria
```
kubectl get pods | grep Evicted | awk '{print $1}' 
```

### Delete pods by criteria
```
kubectl get pods | grep Evicted | awk '{print $1}' | xargs kubectl delete pod
```
