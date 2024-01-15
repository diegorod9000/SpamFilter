import csv

DATASET = "data/spam_ham_dataset.csv"
DIR_CLEAN = "data/clean_large"
DIR_SPAM = "data/spam_large"
DIR_TEST = "data/testing_large"

if __name__ == "__main__":
    csvLength = 0
    with open(DATASET) as csvFile:
        spamReader_len = csv.reader(csvFile, delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csvLength = sum(1 for row in spamReader_len)
        
    with open(DATASET) as csvFile:
        spamReader = csv.reader(csvFile, delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        split_index = int(csvLength*0.8)
        print(csvLength)
        i = 0
        for row in spamReader:
            print(row[1])
            if(i<split_index):
                fileName = row[0]
                textContent = row[2]
                fileDir = DIR_CLEAN if row[1]=="ham" else DIR_SPAM
                filePath = "%s/%s.txt"%(fileDir,fileName)
                
                f = open(filePath, "w+")
                f.write(textContent)
                f.close()
            else:
                fileName = row[0]
                textContent = row[2]
                fileExt= row[1]
                filePath = "%s/%s.%s.txt" % (DIR_TEST,fileName,fileExt)

                f = open(filePath, "w+")
                f.write(textContent)
                f.close()
            i+=1
            
            
            
            