# Linux: Cache Git Credentials
Set Git to cache credentials in memory:
```
git config --global credential.helper cache
```

Change the default cache timeout (in seconds):
```
git config --global credential.helper 'cache --timeout=10800'
```

# macOS: Use Keychain for Git Credentials
Check if `osxkeychain` helper is installed:
```
git credential-osxkeychain
```

Set Git to use the `osxkeychain` helper:
```
git config --global credential.helper osxkeychain
```

# Reset Credentials for One Repository
Clear local credential helper so Git asks again for credentials:
```
git config --local credential.helper ""
```

# Reset Local Branch to Match Remote Branch
Use this when you want local files to exactly match `origin/v3`:
```
git fetch origin
git reset --hard origin/v3
git clean -fd
```

# Merge a branch into `master`
1. Make sure your local repo is up to date.
```
git fetch origin
```

2. Switch to `master`.
```
git checkout master
```

3. Pull latest `master`.
```
git pull origin master
```

4. Merge `v3` into `master`.
```
git merge v3
```

5. Push updated `master`.
```
git push origin master
```

If Git opens Vim during merge commit, save and quit with:
```
:wq
```

# Create a New Branch from Updated `master`
```
git switch master
git pull origin master
git switch -c v4-hiding-post
```

Push this branch after creating it.

# Start `v7` from `v6`
```
git checkout v6
git checkout -b v7
```

# Merge Completed `v7` Back to `v6`
```
git checkout v6
git merge v7
```

# Merge `v6` into `master`
```
git checkout master
git merge v6
```

# Update `v7` with Latest Changes from `v6`
If you pushed new commits to `v6` and want `v7` to match:
```
git checkout v7
git merge v6
```
