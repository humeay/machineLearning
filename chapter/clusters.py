
def readfile(filename):
    lines = []
    with open(filename,'r') as f:
        for line in f:
            lines.append(line)

    colnames = lines[0].strip().split('\t')[1:]
    rownames = []
    data = []
    for line in lines[1:]:
        p = line.strip().split('\t')
        rownames.append(p[0])
        data.append([float(x) for x in p[1:]])
    return rownames,colnames,data

if __name__ == '__main__':
    mylist = readfile('blogdata.txt')
    print(mylist[0])
    print(mylist[1])
    print(mylist[2])