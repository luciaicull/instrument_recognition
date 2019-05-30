from datetime import datetime

def write_result(test_num, model_name, parameters, acc, class_acc, comments):
    file_path = './test_result/Test Result '+str(test_num)+'.txt'
    f = open(file_path, 'w')

    note = []
    note.append("Test number: "+str(test_num))
    note.append("Test day: "+str(datetime.now()))
    note.append("============================================")

    note.append("Model: "+model_name)

    note.append("Parameters")
    pa = parameters.split('+ ')
    for param in pa:
        note.append('\t'+param)

    note.append("Test Accuracy")
    note.append('\t'+str(acc))

    note.append("\t~ Class Accuracy ~ ")
    inst = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'vocal']
    for i in range(0,len(inst)):
        note.append('\t\t' + inst[i] + '    ' + str(class_acc[i]) + '%')

    note.append("Comments")
    note.append(comments)

    for message in note:
        f.write(message + '\n')

    f.close()
