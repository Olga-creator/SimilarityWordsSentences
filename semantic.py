# Import libraries
import spacy

# spacy.cli.download("en_core_web_md")
# Specifying the model we want to use
nlp = spacy.load("en_core_web_md")

# Determine similarity between words
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))
print()

"""
Write a note about what you found interesting about the similarities between cat, monkey and banana 
and think of an example of your own:
    1) Cat and monkey seem to be similar because they are both animals;
    2) Interestingly, monkey and banana have a higher similarity than cat and banana. So we can assume that the model 
    already puts together that monkeys eat bananas and that is why there is a significant similarity.
    3) Another interesting fact is that cat does not have significant similarity with banana although monkey does. 
    So, the model does not explicitly seem to recognise transitive relationships in its calculation.
"""

# Another example of similarity between words
word1 = nlp("boat")
word2 = nlp("ocean")
word3 = nlp("fish")

print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

"""
Write a note about what you found interesting about the similarities in your own example 
(similarity between boat, ocean and fish):
    1) Interestingly, boat and ocean have higher similarity than fish and ocean but both pairs of words have very close
    the cosine similarity. So we can assume that the model already puts together that boats sail on an ocean and 
    fish live in the ocean that is why there is a significant similarity.
    2) Another interesting fact is that boat and fish have a high similarity. So we can assume that the model already 
    puts together that both fish and boat have a streamlined body that lowers the friction which helps to move with 
    less effort that is why there is a high similarity but not as significant as between boat - ocean and fish - ocean.
"""


# Compare series of words with one another
tokens = nlp("cat apple monkey banana")

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))


# Compare a sentence with a list of sentences and calculate the cosine similarity between them
sentence_to_compare = "Why is my cat on the car"

sentences = ["where did my dog go",
             "Hello, there is my car",
             "I\'ve lost my car in my car",
             "I\'d like my boat back",
             "I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)

for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)


"""
Question: Run the example file with the simpler language model ‘en_core_web_sm’ and write a note on what you notice 
is different from the model 'en_core_web_md'

Answer: Running the example file with model 'en_core_web_md' gives significantly higher similarity than when file 
is run using the simpler language model ‘en_core_web_sm’. The difference is in the accuracy of the predictions.
"""
