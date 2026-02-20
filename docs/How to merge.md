## How to merge the main branch into your feature branch

1. Make sure you are on your feature branch, else run
```git checkout my-branch```

2. If you have uncommitted changes, you must commit them first

3. ```git fetch origin```

4. ```git merge origin/main```

5. You may now have merge conflicts. Resolve them using the VS Code interface. After you have resolved a file, click the blue button in the bottom right "Complete merge" (or something). Do that for all conflicting files.

6. Commit your merged files.

7. Push to origin

8. Done!