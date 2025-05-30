import random
import re


# Claire Gillaspie
# 2/3/25
# CMSC 437
# ELIZA Project

### DESCRIPTION & HOW TO RUN ###
# This program creates a psychotherapist chatbot named Celiza.
# Celiza will listen and respond to a user's problems.
# Runs on the command line with no arguments: python eliza.py

### ALGORITHM ###
# Reads in current line of user input
# Parses input by whitespace, checks if each word is a "keyword"
# If current word is a keyword, check ranking & replace current keyword if ranking is higher
# At the end of the input, the decomposition rules for the highest ranking keyword are evaluated
# Finally, a reassembly rule is chosen to produce output, and the cycle continues
# Decomposition and reconstruction rules use regex to determine context

### SAMPLE INPUT & OUTPUT ###
# Hi, I'm Celiza. What's your name?
# hi my name is mayra nice to meet you
# Hi, Mayra. What would you like to discuss?
# how do i get into med school
# Are there steps you can take to learn how to get into med school?
# yes but its hard and i dont know what to do
# Can you ask someone you know for help?
# i dont know anyone to ask
# Why dont you know anyone to ask?
# everyone just has different opinions
# How does that make you feel?
# it makes me feel nervous and scared
# Mayra, why do you feel nervous and scared?
# bye i have to go
# Goodbye, have a nice day


# Default outputs when no keyword is recognized
def confused():
    return output_randomizer(["How does that make you feel?\n",
                              "Tell me more...\n",
                              "Does that bother you?\n",
                              "What else is on your mind?\n",
                              "That sounds hard\n",
                              "I am not sure I understand you\n",
                              "Your feelings are valid\n",
                              "Take it one step at a time\n",
                              "Don't let worries about the future take away from the joy of the present\n"])


# Selects a random output from given options
def output_randomizer(inputs):
    i = random.randint(0, len(inputs) - 1)
    # select a random  output
    random_output = inputs[i]

    return random_output


# Initializes dictionary of keywords & their ranking
def dictionary():
    keywords = {
        "BYE": 11,
        "NAME": 11,
        "FEEL": 9,
        "FEELING": 9,
        "FELT": 9,
        "BECAUSE": 5,
        "DON'T": 8,
        "DONT": 8,
        "DIDN'T": 8,
        "DIDNT": 8,
        "MY": 5,
        "I'M": 4,
        "IM": 4,
        "AM": 4,
        "HAVE": 4,
        "HAD": 4,
        "HAVEN'T": 4,
        "HAVENT": 4,
        "LAZY": 9,
        "TIRED": 9,
        "WORK": 7,
        "BIOCHEM": 10,
        "CHEMISTRY": 10,
        "CHEM": 10,
        "BIOCHEMISTRY": 10,
        "MCAT": 10,
        "REMEMBER": 8,
        "WANT": 9,
        "HOW": 7,
        "WHAT": 7
    }
    return keywords


# Uses regex to find appropriate decomposition rule & apply associated reassembly rule based on highest ranking keyword
def construct_output(word, user_response, name):
    # replace first-person pronouns with second-person equivalents
    user_response = re.sub(r"\b[iI]\s*('?m|[Aa]m)\b", "you're", user_response)
    user_response = re.sub(r"\b[Mm]y\b", "your", user_response)
    user_response = re.sub(r"\b([iI]|[Mm]e|[Ww]e)\b", "you", user_response)
    user_response = re.sub(r"\b[Mm]yself\b", "yourself", user_response)
    if word is None:
        return confused()
    if word == "MY":
        # match '___ your [mom][dad][brother][sister][family] ___'
        # (OG input: '___ my [mom][dad][brother][sister][family] ___')
        match1 = re.search(r"\byour\s+([Mm]om|[Dd]ad|[Bb]rother|[Ss]ister|[Ff]amily)\b", user_response)
        if match1:
            return output_randomizer(("Tell me more about your " + match1.group(1).lower() + "\n",
                                      "Are you close with your " + match1.group(1).lower() + "?\n",
                                      "How does your " + match1.group(1).lower() + " make you feel?\n",
                                      "What's your relationship with your " + match1.group(1).lower() + " like?\n"))
        # match '___ your [one-word]' (OG input: '___ my [one-word]'
        match2 = re.search(r"\byour\s+(\w*$)", user_response)
        if match2:
            return output_randomizer(("What do you think about your " + match2.group(1) + "?\n",
                                      "How do you feel about your " + match2.group(1) + "?\n",
                                      "Why do you bring up your " + match2.group(1) + "?\n",
                                      "Describe your " + match2.group(1) + " more\n"))
        else:
            return confused()
    if (word == "I'M") | (word == "IM") | (word == "AM"):
        # match ___ [am] you ___ (OG input: ___ [am] I ___)
        match1 = re.search(r"\b[Aa]m\s+you\b(.*)", user_response)
        if match1:
            return output_randomizer(("Why do you want to know if you're" + match1.group(1) + "?\n",
                                      "Do you often wonder if you might be" + match1.group(1) + "?\n",
                                      "Would you like to be" + match1.group(1) + "?\n",
                                      "Do other people think you're" + match1.group(1) + "?\n"))
        # match ___ you're ___ (OG input: ___ [I'm][I am] ___)
        match2 = re.search(r"\byou're\b(.*)", user_response)
        if match2:
            # remove additional 'I am' ('you're') - ex. 'I am sad, and I am lonely' becomes 'Why are you sad and lonely'
            edited_match = re.sub("you're ", "", match2.group(1))
            return output_randomizer(("Tell me more. Why are you" + edited_match + "?\n",
                                      "Do you prefer to be" + edited_match + "?\n",
                                      "Do other people think you're" + edited_match + "?\n"))
        else:
            return confused()
    if (word == "FEEL") | (word == "FEELING") | (word == "FELT"):
        # match '___ [feel][feeling][felt] ___'
        match = re.search(r"\b[Ff]ee?l(ing)?t?\b(.*)", user_response)
        if match:
            if name is not None:
                return output_randomizer((name + ", why do you feel" + match.group(2) + "?\n",
                                          name + ", tell me more about how you feel" + match.group(2) + "\n",
                                          name + ", do you usually feel" + match.group(2) + "?\n"))
            else:
                return output_randomizer(("Why do you feel" + match.group(2) + "?\n",
                                          "Tell me more about how you feel" + match.group(2) + "\n",
                                          "Do you usually feel" + match.group(2) + "?\n"))
        else:
            return confused()
    if (word == "LAZY") | (word == "TIRED"):
        return output_randomizer(('Dont be a lazy bum\n',
                                  'Okay.. sounds like an excuse\n',
                                  'Nobody wants to work these days\n'))
    if word == "WORK":
        return 'Lazy people should never laugh.. get your work done\n'
    if (word == "BIOCHEM") | (word == "BIOCHEMISTRY") | (word == "CHEMISTRY") | (word == "CHEM"):
        # matches '___ [biochem][biochemistry][bio chem][bio chemistry] is ___'
        match = re.search(r"\b([Bb]io\s*[Cc]hem(istry)?\s+is.*)", user_response)
        if match:
            return output_randomizer(("I don't care if " + match.group(1).lower() + ", you still need to get it done\n",
                                      "Saying " + match.group(1).lower() + " sounds like an excuse...\n"))
        else:
            return "Don't let biochem win\n"
    if word == "MCAT":
        return output_randomizer(("The MCAT should be scared of you\n",
                                  "Think of how nice it will feel to be done\n",
                                  "Don't worry too much about the MCAT, just take it one step at a time\n",
                                  "The MCAT is hard but you've got this\n"))
    if word == "REMEMBER":
        # match ___ [don't][can't] remember ___
        match1 = re.search(r"\b([Dd]on'?t|[Cc]an'?t) [Rr]emember\b(.*)", user_response)
        if match1:
            return "Why " + match1.group(1).lower() + " you remember" + match1.group(2) + "?\n"
        # match ___ remember ___
        match2 = re.search(r"\b[Rr]emember\b(.*)", user_response)
        if match2:
            return output_randomizer(("You remember" + match2.group(1),
                                      "Do you often think of" + match2.group(1) + "?\n",
                                      "Does thinking of" + match2.group(1) + " bring anything else to mind?\n",
                                      "What else do you remember?\n"))
        else:
            return confused()
    if word == "WANT":
        # match ___ don't want ___
        match = re.search(r"\b[dD]on'?t\s+[wW]ant\b(.*)", user_response)
        if match:
            if name is not None:
                return name + ", why do you not want" + match.group(1) + "?\n"
            else:
                return "Why do you not want" + match.group(1) + "?\n"
        else:
            if name is not None:
                return output_randomizer((name + ", why is that what you want?\n",
                                          name + ", tell me more about what you want.\n"))
            else:
                return output_randomizer(("Why is that what you want?\n",
                                          "Tell me more about what you want\n"))
    if (word == "HOW") | (word == "WHAT"):
        # match ___ [what do you do about][what can you do about][what should you do about] ___
        # (OG input: ___ [what do I do about][what can I do about][what should I do about] ___'
        match1 = re.search(r"\b([Ww]hat\s+(?:do|can|should)\s+you\s+do\s+about)\b(.*)", user_response)
        if match1:
            return "Why do you want to know what to do about" + match1.group(2) + "?\n"
        # match ___ [how do you][what do you do][what can you do][what should you do] ___
        # (OG input: ___ [how do I][what do I do][what can I do][what should I do] ___'
        match2 = re.search(r"\b([Hh]ow\s+do\s+you|[Ww]hat\s+(?:do|can|should)\s+you\s+do)\b(.*)", user_response)
        if match2:
            # finds text after the last occurrence of 'to' (occurrence of 'to' not followed by 'to')
            # ex. 'How do I learn how to move on' becomes 'Why do you want to know how to move on?'
            # Instead of 'Why do you want to know how to learn how to move on?'
            edited_match2 = re.search(r"\bto(?!.*\bto)(.*)", match2.group(2))
            if edited_match2:
                return output_randomizer(("Why do you want to know how to" + edited_match2.group(1) + "?\n",
                                          "Are there steps you can take to learn how to" + edited_match2.group(
                                              1) + "?\n"))
            else:
                # does not contain an instance of 'to'
                return output_randomizer(("Why do you want to know how to" + match2.group(2) + "?\n",
                                          "Are there steps you can take to learn how to" + match2.group(2) + "?\n"))
        else:
            return "Can you ask someone you know for help?\n"
    if (word == "DON'T") | (word == "DONT") | (word == "DIDN'T") | (word == "DIDNT"):
        # match '___ [don't][didn't] ___'
        match = re.search(r"\b([Dd]on'?t|[Dd]idn'?t)\b(.*)", user_response)
        if match:
            return output_randomizer(("Why " + match.group(1) + " you" + match.group(2) + "?\n",
                                      "Do you wish you could" + match.group(2) + "?\n"))
        else:
            return confused()
    if (word == "HAVE") | (word == "HAD") | (word == "HAVEN'T") | (word == "HAVENT"):
        # match ___ you have ___ (OG input: I have)
        match_have = re.search(r"you\s+[Hh]ave\b(.*)", user_response)
        if match_have:
            return "Why do you have" + match_have.group(1) + "?\n"
        # match ___ you haven't ___ (OG input: I haven't)
        match_havent = re.search(r"you\s+[Hh]aven'?t\b(.*)", user_response)
        if match_havent:
            return "Why haven't you" + match_havent.group(1) + "?\n"
        # match ___ you had ___ (OG input: I had)
        match_had = re.search(r"you\s+[Hh]ad\b(.*)", user_response)
        if match_had:
            return "Why did you have" + match_had.group(1) + "?\n"
        else:
            return confused()
    if word == "BECAUSE":
        # match '___ because ___'
        match = re.search(r"\b[Bb]ecause\b(.*)", user_response)
        if match:
            # remove additional 'because', ex. 'because I need money and because I'm sad' becomes
            # 'Why do you think you need money and you're sad?'
            edited_match = re.sub("because ", "", match.group(1))
            # ex. 'Because I was hungry' becomes 'Why do you think you were hungry?'
            edited_match = re.sub("was", "were", match.group(1))
            return "Why do you think" + edited_match + "?\n"
        else:
            return confused()


## Finds top keyword in user's input
def find_keyword(user_response):
    # initialize keywords & rankings
    list_of_keywords = dictionary()

    # split input by whitespace
    parsed_input = user_response.split()

    # initial keyword metadata
    highest_rank_keyword = None
    highest_rank = 0

    # determine if each word in input is a keyword
    for w in parsed_input:
        # if current word is a keyword
        if w in list_of_keywords:
            # and the ranking is higher than the ranking of any other keyword found
            if list_of_keywords[w] > highest_rank:
                # update keyword metadata
                highest_rank_keyword = w
                highest_rank = list_of_keywords[w]

    # return most important keyword
    return highest_rank_keyword


if __name__ == "__main__":
    # initial set-up
    run = True
    current_name = None
    output = "Hi, I'm Celiza. What's your name?\n"

    # collect input
    while run:
        response = input(output)
        # remove all punctuation (need to ID all keywords w/o accounting for punctuation)
        response = re.sub("[?,.!]", "", response)
        # find the highest ranking keyword
        keyword = find_keyword(response.upper())
        # store user's name
        if keyword == "NAME":
            name_match = re.search(r"name is (\w+)", response)
            if name_match:
                current_name = name_match.group(1).capitalize()
                output = "Hi, " + current_name + ". What would you like to discuss?\n"
            else:
                output = "I'm sorry, I didn't catch that. If you want to try again, use the format 'My name is ___'.\n"
        # exit the program if user types 'bye'
        elif keyword == "BYE":
            print('Goodbye, have a nice day\n')
            run = False
        # generate appropriate response based on keyword & context
        else:
            output = construct_output(keyword, response, current_name)
