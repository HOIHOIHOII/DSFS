#encoding= utf-8
from __future__ import division
from collections import Counter
from collections import defaultdict

users = [
{ "id":0 , "name" : "Hero"},
{ "id":1 , "name" : "Dunn"},
{ "id":2 , "name" : "Sue"},
{ "id":3 , "name" : "Chi"},
{ "id":4 , "name" : "Thor"},
{ "id":5 , "name" : "Clive"},
{ "id":6 , "name" : "Hicks"},
{ "id":7 , "name" : "Devin"},
{ "id":8 , "name" : "Kate"},
{ "id":9 , "name" :
 "Klein"}
]

friendships = [(0,1),(0,2),(1,2),(1,3),
                (2,3),(3,4),(4,5),(5,6),
                (5,7),(6,8),(7,8),(8,9)]

for user in users:
        user["friends"] = []

for i,j in friendships:
    users[i]["friends"].append(users[j]["id"])
    users[j]["friends"].append(users[i]["id"])

def number_of_friends(user):
    return len(user["friends"])

total_connections = sum(number_of_friends(user) for user in users)



num_users = len(users)
avg_connections = total_connections/ num_users

num_friends_by_id = [(user["id"], number_of_friends(user)) for user in users]

num_friends_by_id = sorted(num_friends_by_id, key=lambda (user_id, num_friends): num_friends, reverse=True)


# def friends_of_friend_ids_bad(user):
    # "foaf" is short for "friend of a friend"
    # return [foaf["id"] #foaf is dict
            # for friend in user["friends"]#friend is string,
            # for foaf in friend["friends"]]#



def get_friend_ids(list_of_users):
    foaf_ids = [foaf_id
                    for friend in users 
                        for foaf_id in friend["friends"]
                            if friend["id"] in list_of_users
                 ]
    return foaf_ids

A = get_friend_ids(users[3]["friends"])
# print A
# print Counter(A)
# print Counter(A).keys()


interests = [
    (0, "Hadoop"), (0, "Big Data"), (0, "HBase"), (0, "Java"),
    (0, "Spark"), (0, "Storm"), (0, "Cassandra"),
    (1, "NoSQL"), (1, "MongoDB"), (1, "Cassandra"), (1, "HBase"),
    (1, "Postgres"), (2, "Python"), (2, "scikit-learn"), (2, "scipy"),
    (2, "numpy"), (2, "statsmodels"), (2, "pandas"), (3, "R"), (3, "Python"),
    (3, "statistics"), (3, "regression"), (3, "probability"),
    (4, "machine learning"), (4, "regression"), (4, "decision trees"),
    (4, "libsvm"), (5, "Python"), (5, "R"), (5, "Java"), (5, "C++"),
    (5, "Haskell"), (5, "programming languages"), (6, "statistics"),
    (6, "probability"), (6, "mathematics"), (6, "theory"),
    (7, "machine learning"), (7, "scikit-learn"), (7, "Mahout"),
    (7, "neural networks"), (8, "neural networks"), (8, "deep learning"),
    (8, "Big Data"), (8, "artificial intelligence"), (9, "Hadoop"),
    (9, "Java"), (9, "MapReduce"), (9, "Big Data")
]

def data_scientists_who_like(target_interest):
    return [user_id
            for user_id, user_interest in interests
            if user_interest == target_interest]

user_id_by_interest = defaultdict(list)
interests_by_user_id = defaultdict(list)

for user_id, interest in interests:
    user_id_by_interest[interest].append(user_id)
    interests_by_user_id[user_id].append(interest)

# print user_id_by_interest
# print interests_by_user_id


def most_common_interests_with(user):
    return Counter(interested_user_id 
        for interests in interests_by_user_id[user["id"]]
        for interested_user_id in user_id_by_interest[interest]
            if interested_user_id != user["id"])

# print most_common_interests_with(users[0])

# print interests


salaries_and_tenures = [(83000, 8.7), (88000, 8.1),
                        (48000, 0.7), (76000, 6),
                        (69000, 6.5), (76000, 7.5),
                        (60000, 2.5), (83000, 10),
                        (48000, 1.9), (63000, 4.2)]

salary_by_tenure = defaultdict(list)

for salary, tenure in salaries_and_tenures:
    salary_by_tenure[tenure].append(salary)

# print salary_by_tenure

average_salary_by_tenure = {
    tenure: sum(salaries) / len(salaries)
    for tenure, salaries in salary_by_tenure.items()
}

# print average_salary_by_tenure

def tenure_bucket(tenure):
    if tenure < 2:
        return "less than two"
    elif tenure < 5:
        return "between two and five"
    else:
        return "more than five"

salary_by_tenure_bucket = defaultdict(list)

for salary, tenure in salaries_and_tenures:
    bucket = tenure_bucket(tenure)
    salary_by_tenure_bucket[bucket].append(salary)

# print salary_by_tenure_bucket


average_salary_by_bucket = {
    tenure_bucket : sum(salaries)/ len(salaries)
    for tenure_bucket, salaries in salary_by_tenure_bucket.items()
}

# print average_salary_by_bucket
A = Counter(
'''After ascending a crescent shaped, red-carpeted stair you  among white marble statues, you 
enter the dazzling main hall of Piergeiron’s Palace. The vaulted ceiling rises high above you and beneath it, 
floating unsupported are several massive crystal chandeliers, shining a pure white light. A three huge windows 
made of single plates of crystal are curtained in the finest white gauze and by them immense stone amphoras of 
the most exotic wines. Stewards carrying huge plates of food are moving slowly among the crowd of assembled nobles. 
In a semi circle against the far wall in the centre of the room, there is a sectioned area which separates those of 
lesser note from the great. Against the wall under a golden canopy is the throne of Piergeiron, and on it rests his 
huge mace of office. Manturov is already talking to several impressively bejeweled people, one of whom is completely 
covered in a heavy black robe with black gloves, wearing a glassy obsidian mask. From reputation, you recognise this 
as one of the masked lords of Waterdeep.'''.replace("\n","").split(" ")
)

for word, count in A.most_common(10):
    print word, count