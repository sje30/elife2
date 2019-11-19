# elife2

Second example, python, from
<https://elifesciences.org/articles/44753>


## source code 1 (py)

No no no... none of these files.

```
home_folder = '/Users/Unimusic/Desktop/Smith_Lab'
label_folder = os.path.join(home_folder, 'flipped prs 10.3.18')
gt_file = os.path.join(home_folder, 'REformatted Nuclear coordinates.xlsx')
```
	

## source code 2 (py)

same worries

```
''' ----------- start algo ----------- '''

home_folder = '/Users/Unimusic/Desktop/Smith_Lab'
label_folder = os.path.join(home_folder, 'flipped rns 10.4.18')
gt_file = os.path.join(home_folder, 'REformatted Nuclear coordinates.xlsx')
```

## source code 3 (py)

Same problems (also, what is credentials file?).

```
plotly.tools.set_credentials_file(username='unimusic',
api_key='I1TfBXlucguG0uXh9lPV')
...
home_folder = '/Users/Unimusic/Desktop/Smith_Lab'
label_folder = os.path.join(home_folder, 'flipped prs 10.3.18')
gt_file = os.path.join(home_folder, 'all cells - formatted for angela.xlsx')
...
	currfolder = os.path.join(label_folder, subfolder)
	reg_file = os.path.join(currfolder, 'All_cells.xls')
	if not os.path.exists(reg_file):
		reg_file = os.path.join(currfolder, 'All_cells.xlsx')
	affine_name = os.path.join(currfolder, 'Affine_Matrix.mat')
```
## source code 4 (matlab)

`elife-44753-code4-v2.txt` is a matlab file, that processes TIFF
files.  Where are those TIFF files?

