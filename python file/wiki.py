import pandas as pd
import wikipedia as wp
sports_list = ["sport","cricket","football","basketball","wrestling","kabaddi","hockey","Game"]
politics_list = ["Politics","Politician","Democratic republic","Legislature","Parliament","Election","Government","President (government title)","Policy","Republic","Governance","Prime Minister of India"]
entertainment_list =["Entertainment","music","song","movie","Comedy","film","circus"]
education_list = ["Education","Mathematics","Chemistry","Course (education)","Student","Test (assessment)","Teacher"]
terrorism_list = ["Terrorism","War","Violence","Jihad","Al-Qaeda"]

def get_data(parameter):
    var_data = []
    for i in parameter:
        data1 = wp.page(i)
        s = ""
        for character in data1.content:
            s = s + character
            if(character == '.' or character == '\n'):
                if (len(s) >= 60):
                    var_data.append(s)
                s = ""
    return var_data

list1 = get_data(sports_list)
list4 = get_data(politics_list)
list5 = get_data(terrorism_list)

df1 = pd.DataFrame(list1)
df1.to_csv("1sports_csv.csv",index=False)
# df2 = pd.DataFrame(list2)
# df2.to_csv("2enter_csv.csv",index=False)
# df3 = pd.DataFrame(list3)
# df3.to_csv("3education_csv.csv",index=False)
df4 = pd.DataFrame(list4)
df4.to_csv("4politics_csv.csv",index=False)
df5 = pd.DataFrame(list5)
df5.to_csv("5terrorism_csv.csv",index=False)

data1 = pd.read_csv("1sports_csv.csv")
data1["category"] = "sports"
data1.rename(columns={'0': 'data'}, inplace=True)
print(data1)

data4 = pd.read_csv("4politics_csv.csv")
data4["category"] = "politics"
data4.rename(columns={'0': 'data'}, inplace=True)
print(data4)

data5 = pd.read_csv("5terrorism_csv.csv")
data5["category"] = "terrorism"
data5.rename(columns={'0': 'data'}, inplace=True)
print(data5)

combined1_csv = pd.concat([data1,data4,data5])
print(len(data1),len(data4),len(data5))
combined1_csv.to_csv( "combined1_csv.csv", index=False, encoding='utf-8-sig')
dff=pd.read_csv('combined1_csv.csv')
print(dff)
