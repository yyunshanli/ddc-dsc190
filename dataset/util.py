import os

def ez_name(x):
    """
    Cleans and formats a given string x by removing spaces, replacing non-alphanumeric characters with underscores (_), 
    and ensuring it returns a string suitable for naming files or directories.
    _______
    ez_name("Hello World! 2023")  # Returns 'HelloWorld_2023'
    """
    x = ''.join(x.strip().split())
    x_clean = []
    for char in x:
        if char.isalnum():
            x_clean.append(char)
        else:
            x_clean.append('_')
    return ''.join(x_clean)

def get_subdirs(root, choose=False):
    """
    Lists the names of all subdirectories within a given root directory. 
    Optionally, it allows the user to select specific subdirectories to return.
    """
    subdir_names = sorted([x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x))])
    if choose:
        for i, subdir_name in enumerate(subdir_names):
            print('{}: {}'.format(i, subdir_name))
        subdir_idxs = [int(x) for x in input('Which subdir(s)? ').split(',')]
        subdir_names = [subdir_names[i] for i in subdir_idxs]
    return subdir_names