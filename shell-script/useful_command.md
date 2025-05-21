## Linux Commands
Give access of a folder to all users in Linux:
```chmod -R a+rwx folder/```

Change name of all files in a directory using format:

```ls | cat -n | while read n f; do mv "$f" $(printf "%06d.png" "$n"); done```

To find several files (in the example, PNG images) with specific name format and do whatever you want to do with them (in the example, convert them to JPG images), use the following example which uses command line tool find:

```
find . -type f -name "*.png" -not -name "*heatmaps.png" -print | while read f; do
  mogrify -format jpg $f
done
```

Delete folders inside the path which name includes name_desired:

```
rm -rf `find . -maxdepth <num> -type d -name "<name_desired>"`
```

Find 10 random files:
```
ls *.pdf | gshuf | head -10
```

Count the number of files in a folder:
```
ls <folder_path> | wc -l
```

Count all files within a directory and its subdirectories:
```
find . -type f | wc -l
find ./ -type f -iname "*.JPEG" | wc -l
```

To see which processes are running:
```
ps aux | grep python
```

### Remote Commands
Download small file from google drive:
```
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O FILENAME
```

Download COCO:
```
https://gist.github.com/mkocabas/a6177fc00315403d31572e17700d7fd9
```

## Python Commands
Install pycocotools:
```
conda install -c conda-forge pycocotools
```
Encountering a GPU leak in a Python program:
```
import signal, os
os.kill(<PID>, signal.SIGKILL)
```
