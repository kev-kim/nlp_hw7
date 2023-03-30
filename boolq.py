import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

qa_pairs = [
    ("does ethanol take more energy make that produces", "false"),
    ("is house tax and property tax are same", "true"),
    ("is pain experienced in a missing body part or paralyzed area", "true"),
    ("is harry potter and the escape from gringotts a roller coaster ride", "true"),
    ("is there a difference between hydroxyzine hcl and hydroxyzine pam", "true"),
    ("is barq's root beer a pepsi product", "false"),
    ("can an odd number be divided by an even number", "true"),
    ("is there a word with q without u", "true"),
    ("can u drive in canada with us license", "true"),
    ("is there a play off for third place in the world cup", "true"),
    ("can minors drink with parents in new york", "true"),
    ("is the show bloodline based on a true story", "false"),
    ("is it bad to wash your hair with shower gel", "true"),
    ("is the liver part of the excretory system", "true"),
    ("is fantastic beasts and where to find them a prequel", "true"),
    ("will there be a season 8 of vampire diaries", "true"),
    ("was the movie strangers based on a true story", "true"),
    ("is durham university part of the russell group", "true"),
    ("is the tv show the resident over for the season", "true"),
    ("does magnesium citrate have citric acid in it", "true"),
    ("does p o box come before street address", "false"),
    ("does a spark plug keep an engine running", "true"),
    ("is a cape and a cloak the same", "true"),
    ("does it cost money to renounce us citizenship", "true"),
    ("is a fire 7 the same as a kindle", "true"),
    ("can you drink alcohol with your parents in wisconsin", "true"),
    ("do penguins have feathers arising from the epidermis", "true"),
    ("do you need to break in a car", "false"),
    ("is the enchanted forest in oregon still open", "true"),
    ("is there a golf course at the indy 500", "true"),
]
fewshot = "Answer just \"true\" or \"false\" for the following questions. The answers must be 100% factually accurate.\n\nQ: in the phantom menace is padme the queen\nA: true\n\nQ: does anyone on instagram have 1 billion followers\nA: false\n\nQ: does age have any influence on attentional ability or inattentional blindness\nA: true\n\nQ: you can tell a lot about a person by how they treat their waiter\nA: true\n\nQ: is wheat flour and white flour the same thing\nA: false\n\nQ: does the cat in the hat have a name\nA: false\n\nQ: do you always have to say check in chess\nA: false\n\nQ: is windows movie maker part of windows essentials\nA: true\n\nQ:"

cumul = 0
for pair in qa_pairs:
    response = openai.Completion.create(
        model="davinci",
        prompt= fewshot + pair[0] + "\nA:",
        temperature=0.7,
    	max_tokens=2,
    	top_p=1,
    	frequency_penalty=0,
    	presence_penalty=0
	)
    if response["choices"][0]["text"].strip() == pair[1]:
        cumul += 1
    else:
        print("WRONG: " + pair[0])

print("Total accuracy:", cumul/30)