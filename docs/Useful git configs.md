```
git config --global push.autoSetupRemote true
git config --global push.default current
git config --global pull.ff only
```` 

Normalize line endings:
Windows:
```
git config --global core.autocrlf true
```

Linux/macos:
```
git config --global core.autocrlf input
```