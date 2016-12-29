import sys
import random
file_table2 = sys.argv[1]
file_table3 = sys.argv[2]
file_table4 = sys.argv[3]
file_table5 = sys.argv[4]

#----------------------------TRAIN-TRAIN-TRAIN--------------------------
#-------------------------------Read-Data-------------------------------
train_array1 = []
with open(file_table2) as my_file:
    for line in my_file:
        line=line.lower()
        line = line.split()
        train_array1.append(line[0:])
#print len(train_array)
add1=float(1.0000)


for i in train_array1:
    t=len(i)

for i in train_array1:
    i.insert(0,add1)
    #print len(i)
#random.shuffle(train_array)
Total_rows=len(train_array1)

#------------------------------Store labels_train-----------------------------
labels_train1=[]
with open(file_table3) as my_file:
    for line in my_file:
        line=line.lower()
        line=line.split()
        labels_train1.append(float(line[0]))

for j,l in enumerate(train_array1):
    l.append(labels_train1[j])
    #print l[257]
#print labels_train

random.shuffle(train_array1)
train_array=[]
labels_train=[]
for i in train_array1:
    train_array.append(i[0:-1])
    labels_train.append(i[-1])



#------------------------------TEST-TEST-TEST--------------------------
#-------------------------------Read-Data-------------------------------
test_array1 = []
with open(file_table4) as my_file:
    for line in my_file:
        line=line.lower()
        line = line.split()
        test_array1.append(line[0:])

add1=float(1.0000)

for i in test_array1:
    t=len(i)

for i in test_array1:
    i.insert(0,add1)
#random.shuffle(test_array)
Total_rows=len(test_array1)
#------------------------------Store labels_test-----------------------------
labels_test1=[]
with open(file_table5) as my_file:
    for line in my_file:
        line=line.lower()
        line=line.split()
        labels_test1.append(float(line[0]))

for j,l in enumerate(test_array1):
    l.append(labels_test1[j])
    #print l[257]
#print labels_train

random.shuffle(test_array1)
test_array=[]
labels_test=[]
for i in test_array1:
    test_array.append(i[0:-1])
    labels_test.append(i[-1])

#------------------------------Calculation-------------------------------

gamma0=0.01
epoch_hy=[3,5,8]
C=1
for epoch in epoch_hy:
    bias=random.randint(-1,1)
    weight=random.randint(-1,1)
    weight = [weight]*t
    weight = [bias] + weight
    true=0
    false=0
    #weight1=weight
    for k in range(epoch):
        for j,i in enumerate(train_array):
            i=map(float,i)
            A=map(lambda x,y:x*y,weight,i)
            B=reduce(lambda x,y:x+y,A)
            learning_rate=float(gamma0)/(1+gamma0*((j+1)/C))
            if (labels_train[j]*B)<=1:

                #print "first if"
                D=map (lambda x:x*(1-learning_rate),weight)
                D1=map(lambda x:x*C*labels_train[j]*learning_rate,i)
                D2=map(lambda x,y:x+y,D,D1)
                weight=D2
                true=true+1
            else:
                D=map (lambda x:x*(1-learning_rate),weight)
                weight=D
                false=false+1
        tt=0
        ff=0
    for j,i in enumerate(test_array):
        i=map(float,i)
        EE=map(lambda x,y:x*y,weight,i)
        EE1=reduce(lambda x,y:x+y,EE)
        if EE1*labels_test[j]>0:
            tt+=1
        else:
            ff+=1
    print "Accuracy is", float(tt)/(tt+ff), "Epoch is", epoch, "Random Bias is", bias
