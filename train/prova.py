import csv

# note: If you use 'b' for the mode, you will get a TypeError
# under Python3. You can just use 'w' for Python 3

data=[('smith, bob',2),('carol',3),('ted',4),('alice',5)]

def main():
    with open('ur file.csv','w') as out:
        csv_out=csv.writer(out)
        csv_out.writerow(['name','num'])
        for row in data:
            csv_out.writerow(row)


if __name__ == '__main__':
    main()