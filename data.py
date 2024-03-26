import pymongo
import pandas as pd
# Establish a connection to MongoDB
client = pymongo.MongoClient("mongodb+srv://mulesaikrishnareddy2003:saikris2003@cluster0.bnz2azr.mongodb.net/subject_DB?retryWrites=true&w=majority")


# # Load data from Excel sheet
data = pd.read_excel("major.xlsx")


# Select the database
db = client["subject_DB"]

# Select the collection
collection = db["subjects"]
# Query the collection to retrieve documents with topics that match the provided list
topics_to_search = ["git", "java", "aws"]
matching_documents = collection.find({"topic": {"$in": topics_to_search}})
matching_documents_content=[]
for document in matching_documents:
    # Print the keys present in the content array
    for content_item in document["content"]:
        matching_documents_content.append(content_item['value'])
print(matching_documents_content)
# Iterate over the matching documents and print them
# for document in matching_documents:
#     # Print the keys present in the content array
#     for content_item in document["content"]:
#         print("Topic:",document['topic'],"value:", content_item["value"])

# Initialize dictionary to hold subjects
# subjects_dict = {}

# # Iterate over rows in the dataframe and populate subjects_dict
# for index, row in data.iterrows():
#     topic = row['Topic']
#     content = row['Text']
#     # If topic doesn't exist in subjects_dict, create it
#     if topic not in subjects_dict:
#         subjects_dict[topic] = []
#     # Append content for each topic
#     subjects_dict[topic].append(content)

# # Insert subjects_dict into the collection
# for topic, content_list in subjects_dict.items():
#     document = {
#         "topic": topic,
#         "content": [{"key": i+1, "value": content} for i, content in enumerate(content_list)]
#     }
#     collection.insert_one(document)

# print("Data inserted successfully into MongoDB collection.")

