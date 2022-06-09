def check_male(sexm):
    if (sexm == "male"):
        male = 1

    elif (sexm == "female"):
        male = 0
        
    return male

def check_female(sexf):
    if (sexf == "male"):
        female = 0

    elif (sexf == "female"):
        female = 1
        
    return female

def check_emc(c):
    if (c == "C"):
        emc = 1
    else:
        emc = 0
    return emc

def check_emq(q):
    if (q == "Q"):
        emq = 1
    else:
        emq = 0
    return emq

def check_ems(s):
    if (s == "S"):
        ems = 1
    else:
        ems = 0
    return ems

def check_flow(flow):
    if (flow == "Low"):
        low = 1
    else:
        low = 0
    return low

def check_fmedium(fmedium):
    if (fmedium == "Medium"):
        medium = 1
    else:
        medium = 0
    return medium

def check_fave(fave):
    if (fave == "Average"):
        ave = 1
    else:
        ave = 0
    return ave

def check_fhigh(fhigh):
    if (fhigh == "High"):
        high = 1
    else:
        high = 0
    return high

def survival(predict):
    if (predict == 0):
        sur = "Dead"
    
    elif (predict == 1):
        sur = "Survived"
    
    return sur