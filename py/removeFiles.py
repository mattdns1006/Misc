import os, glob 

def removeFiles(path,check=1):

    assert check in (0,1), "Check must be in {0,1}"
    print("Check = 1 by default which prints files to be deleted in {0}.".format(path))
    files = glob.glob(path+"*")
    count = 0
    for f in files: 
        if check == 1:
            print(f)
        elif check == 0:
            os.remove(f)
        count +=1
    if check == 1:
        print("{0} files to be deleted.".format(count))
    elif check ==0:
        print("{0} files deleted.".format(count))
