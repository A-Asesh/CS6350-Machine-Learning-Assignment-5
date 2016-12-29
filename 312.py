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

#print len(train_array)
#print split_value
part1=[]
part2=[]
part3=[]
part4=[]
part5=[]
split1=[]
split2=[]
split3=[]
split4=[]
split5=[]
#print train_array[0]
for i in range(0,1):
    part1.extend(train_array[0:400])
    part2.extend(train_array[400:800])
    part3.extend(train_array[800:1200])
    part4.extend(train_array[1200:1600])
    part5.extend(train_array[1600:2000])
    split1.extend(train_array[400:2000])
    split2.extend(train_array[800:2000]+train_array[0:400])
    split3.extend(train_array[1200:2000]+train_array[0:800])
    split4.extend(train_array[1600:2000]+train_array[0:1200])
    split5.extend(train_array[0:1600])

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
random.shuffle(test_array1)
test_array=[]
labels_test=[]
for i in test_array1:
    test_array.append(i[0:-1])
    labels_test.append(i[-1])

#------------------------------Calculation 1-------------------------------
epoch_hy=[3,5,8]
value_gamma=[100,0.1,0.001,1]
value_C=[1,0.5,0.25,0.125,0.0625,0.0312]
for epoch in epoch_hy:
    bias=random.randint(-3,3)
    weight=random.randint(-1,1)
    weight = [weight]*t
    weight = [bias] + weight
    true=0
    false=0
    #weight1=weight
    for k in range(epoch):
        for kk2 in value_gamma:
            for kk in value_C:
                for j,i in enumerate(split1):
                    i=map(float,i)
                    A=map(lambda x,y:x*y,weight,i)
                    B=reduce(lambda x,y:x+y,A)
                    learning_rate=float(kk2)/(1+kk2*((j+1)/kk))
                    if (labels_train[j]*B)<=1:

                        #print "first if"
                        D=map (lambda x:x*(1-learning_rate),weight)
                        D1=map(lambda x:x*kk*labels_train[j]*learning_rate,i)
                        D2=map(lambda x,y:x+y,D,D1)
                        weight=D2
                        true=true+1
                    else:
                        D=map (lambda x:x*(1-learning_rate),weight)
                        weight=D
                        false=false+1
                tt=0
                ff=0
                TP=0
                FP=0
                FN=0
            for j,i in enumerate(part1):
                i=map(float,i)
                EE=map(lambda x,y:x*y,weight,i)
                EE1=reduce(lambda x,y:x+y,EE)
                #print EE1
                if EE1*labels_test[j]>0:
                    tt+=1
                else:
                    ff+=1
                if EE1>0 and labels_test[j]>0:
                    TP+=1
                if EE1>0 and labels_test[j]<0:
                    FP+=1
                if EE1<0 and labels_test[j]>0:
                    FN+=1
            #print TP
            #print FP
            #print FN
            #p=float(TP)/(TP + FP)
            #r=float(TP)/(TP + FN)
            #f1=float (2)*(p*r)/(p+r)
            print "Avg Accuracy is", float(tt)/(tt+ff)
            print "Epoch is", epoch
            print "Gamma is", kk2
            print "C is", kk
            print "Bias Value", bias
            #print "Precision is", p, "Epoch is", epoch
            #print "Recall is", r, "Epoch is", epoch
            #print "F1 Score is", f1, "Epoch is", epoch
#------------------------------Calculation 2-------------------------------

for epoch in epoch_hy:
    bias=random.randint(-3,3)
    weight=random.randint(-1,1)
    weight = [weight]*t
    weight = [bias] + weight
    true=0
    false=0
    #weight1=weight
    for kk2 in value_gamma:
        for kk in value_C:
            for k in range(epoch):
                for j,i in enumerate(split2):
                    i=map(float,i)
                    A=map(lambda x,y:x*y,weight,i)
                    B=reduce(lambda x,y:x+y,A)
                    learning_rate=float(kk2)/(1+kk2*((j+1)/kk))
                    if (labels_train[j]*B)<=1:

                        #print "first if"
                        D=map (lambda x:x*(1-learning_rate),weight)
                        D1=map(lambda x:x*kk*labels_train[j]*learning_rate,i)
                        D2=map(lambda x,y:x+y,D,D1)
                        weight=D2
                        true=true+1
                    else:
                        D=map (lambda x:x*(1-learning_rate),weight)
                        weight=D
                        false=false+1
                tt=0
                ff=0
                TP=0
                FP=0
                FN=0
            for j,i in enumerate(part2):
                i=map(float,i)
                EE=map(lambda x,y:x*y,weight,i)
                EE1=reduce(lambda x,y:x+y,EE)
                #print EE1
                if EE1*labels_test[j]>0:
                    tt+=1
                else:
                    ff+=1
                if EE1>0 and labels_test[j]>0:
                    TP+=1
                if EE1>0 and labels_test[j]<0:
                    FP+=1
                if EE1<0 and labels_test[j]>0:
                    FN+=1
            #print TP
            #print FP
            #print FN
            #p=float(TP)/(TP + FP)
            #r=float(TP)/(TP + FN)
            #f1=float (2)*(p*r)/(p+r)
            print "Avg Accuracy is", float(tt)/(tt+ff)
            print "Epoch is", epoch
            print "Gamma is", kk2
            print "C is", kk
            print "Bias Value", bias
            #print "Precision is", p, "Epoch is", epoch
            #print "Recall is", r, "Epoch is", epoch
            #print "F1 Score is", f1, "Epoch is", epoch
#------------------------------Calculation 3-------------------------------
for epoch in epoch_hy:
    bias=random.randint(-3,3)
    weight=random.randint(-1,1)
    weight = [weight]*t
    weight = [bias] + weight
    true=0
    false=0
    #weight1=weight
    for kk2 in value_gamma:
        for kk in value_C:
            for k in range(epoch):
                for j,i in enumerate(split3):
                    i=map(float,i)
                    A=map(lambda x,y:x*y,weight,i)
                    B=reduce(lambda x,y:x+y,A)
                    learning_rate=float(kk2)/(1+kk2*((j+1)/kk))
                    if (labels_train[j]*B)<=1:

                        #print "first if"
                        D=map (lambda x:x*(1-learning_rate),weight)
                        D1=map(lambda x:x*kk*labels_train[j]*learning_rate,i)
                        D2=map(lambda x,y:x+y,D,D1)
                        weight=D2
                        true=true+1
                    else:
                        D=map (lambda x:x*(1-learning_rate),weight)
                        weight=D
                        false=false+1
                tt=0
                ff=0
                TP=0
                FP=0
                FN=0
            for j,i in enumerate(part3):
                i=map(float,i)
                EE=map(lambda x,y:x*y,weight,i)
                EE1=reduce(lambda x,y:x+y,EE)
                #print EE1
                if EE1*labels_test[j]>0:
                    tt+=1
                else:
                    ff+=1
                if EE1>0 and labels_test[j]>0:
                    TP+=1
                if EE1>0 and labels_test[j]<0:
                    FP+=1
                if EE1<0 and labels_test[j]>0:
                    FN+=1
            #print TP
            #print FP
            #print FN
            #p=float(TP)/(TP + FP)
            #r=float(TP)/(TP + FN)
            #f1=float (2)*(p*r)/(p+r)
            print "Avg Accuracy is", float(tt)/(tt+ff)
            print "Epoch is", epoch
            print "Gamma is", kk2
            print "C is", kk
            print "Bias Value", bias
            #print "Precision is", p, "Epoch is", epoch
            #print "Recall is", r, "Epoch is", epoch
            #print "F1 Score is", f1, "Epoch is", epoch
#------------------------------Calculation 4-------------------------------
for epoch in epoch_hy:
    bias=random.randint(-3,3)
    weight=random.randint(-1,1)
    weight = [weight]*t
    weight = [bias] + weight
    true=0
    false=0
    #weight1=weight
    for kk2 in value_gamma:
        for kk in value_C:
            for k in range(epoch):
                for j,i in enumerate(split4):
                    i=map(float,i)
                    A=map(lambda x,y:x*y,weight,i)
                    B=reduce(lambda x,y:x+y,A)
                    learning_rate=float(kk2)/(1+kk2*((j+1)/kk))
                    if (labels_train[j]*B)<=1:

                        #print "first if"
                        D=map (lambda x:x*(1-learning_rate),weight)
                        D1=map(lambda x:x*kk*labels_train[j]*learning_rate,i)
                        D2=map(lambda x,y:x+y,D,D1)
                        weight=D2
                        true=true+1
                    else:
                        D=map (lambda x:x*(1-learning_rate),weight)
                        weight=D
                        false=false+1
                tt=0
                ff=0
                TP=0
                FP=0
                FN=0
            for j,i in enumerate(part4):
                i=map(float,i)
                EE=map(lambda x,y:x*y,weight,i)
                EE1=reduce(lambda x,y:x+y,EE)
                #print EE1
                if EE1*labels_test[j]>0:
                    tt+=1
                else:
                    ff+=1
                if EE1>0 and labels_test[j]>0:
                    TP+=1
                if EE1>0 and labels_test[j]<0:
                    FP+=1
                if EE1<0 and labels_test[j]>0:
                    FN+=1
            #print TP
            #print FP
            #print FN
            #p=float(TP)/(TP + FP)
            #r=float(TP)/(TP + FN)
            #f1=float (2)*(p*r)/(p+r)
            print "Avg Accuracy is", float(tt)/(tt+ff)
            print "Epoch is", epoch
            print "Gamma is", kk2
            print "C is", kk
            print "Bias Value", bias
            #print "Precision is", p, "Epoch is", epoch
            #print "Recall is", r, "Epoch is", epoch
            #print "F1 Score is", f1, "Epoch is", epoch
#------------------------------Calculation 5-------------------------------
for epoch in epoch_hy:
    bias=random.randint(-3,3)
    weight=random.randint(-1,1)
    weight = [weight]*t
    weight = [bias] + weight
    true=0
    false=0
    #weight1=weight
    for kk2 in value_gamma:
        for kk in value_C:
            for k in range(epoch):
                for j,i in enumerate(split5):
                    i=map(float,i)
                    A=map(lambda x,y:x*y,weight,i)
                    B=reduce(lambda x,y:x+y,A)
                    learning_rate=float(kk2)/(1+kk2*((j+1)/kk))
                    if (labels_train[j]*B)<=1:

                        #print "first if"
                        D=map (lambda x:x*(1-learning_rate),weight)
                        D1=map(lambda x:x*kk*labels_train[j]*learning_rate,i)
                        D2=map(lambda x,y:x+y,D,D1)
                        weight=D2
                        true=true+1
                    else:
                        D=map (lambda x:x*(1-learning_rate),weight)
                        weight=D
                        false=false+1
                tt=0
                ff=0
                TP=0
                FP=0
                FN=0
            for j,i in enumerate(part5):
                i=map(float,i)
                EE=map(lambda x,y:x*y,weight,i)
                EE1=reduce(lambda x,y:x+y,EE)
                #print EE1
                if EE1*labels_test[j]>0:
                    tt+=1
                else:
                    ff+=1
                if EE1>0 and labels_test[j]>0:
                    TP+=1
                if EE1>0 and labels_test[j]<0:
                    FP+=1
                if EE1<0 and labels_test[j]>0:
                    FN+=1
            #print TP
            #print FP
            #print FN
            #p=float(TP)/(TP + FP)
            #r=float(TP)/(TP + FN)
            #f1=float (2)*(p*r)/(p+r)
            print "Avg Accuracy is", float(tt)/(tt+ff)
            print "Epoch is", epoch
            print "Gamma is", kk2
            print "C is", kk
            print "Bias Value", bias#print "Precision is", p, "Epoch is", epoch
            #print "Recall is", r, "Epoch is", epoch
            #print "F1 Score is", f1, "Epoch is", epoch
