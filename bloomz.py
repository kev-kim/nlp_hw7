import os
import requests

API_TOKEN = os.getenv("BLOOMZ_API_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/bigscience/bloomz"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

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
    ("does deadpool have a kid in the comics", "true"),
    ("do they still make benson & hedges cigarettes", "true"),
    ("is federal income tax the same as social security", "false"),
    ("is an engine speed sensor the same as a crankshaft sensor", "true"),
    ("is indiana jones temple of doom a prequel", "true"),
    ("is there any next part of avengers infinity war", "true"),
    ("is the toyota highlander on a truck frame", "false"),
    ("is it legal to do a cover of a song", "true"),
    ("can carbon form polar covalent bonds with hydrogen", "false"),
    ("is there a sequel to the movie the golden compass", "false"),
    ("is columbus day a national holiday in the united states", "true"),
    ("are new balance and nike the same company", "false"),
    ("is there an interstate that goes coast to coast", "true"),
    ("is pureed tomatoes the same as tomato sauce", "false"),
    ("can there be a word without a vowel", "true"),
    ("does only the winner get money on tipping point", "true"),
    ("is there such a thing as a turkey vulture", "true"),
    ("has anyone hit a hole in one on a par 5", "true"),
    ("do the jets and giants share a stadium", "true"),
    ("is the us womens soccer team in the world cup", "true"),
    ("can an african team win the world cup", "true"),
    ("can a hammer be used as a weapon", "true"),
    ("do they still have fox hunts in england", "false"),
    ("can you wear short sleeve shirt with asu jacket", "true"),
    ("has wisconsin ever been in the little league world series", "false"),
    ("does damon and elena get together in season 3", "false"),
    ("is there a player in the nfl missing a hand", "true"),
    ("is the other boleyn girl part of a series", "true"),
    ("is there a group called the five heartbeats", "false"),
    ("is mount everest a part of the himalayas", "true"),
    ("can an emt-basic start an iv", "false"),
    ("has no 1 court at wimbledon got a roof", "false"),
    ("has anyone come back from 3-0 in the nba finals", "false"),
    ("do radio waves travel at the speed of light", "true"),
    ("did anyone from the 1980 us hockey team play in the nhl", "true"),
    ("do all triangles have at least two acute angles", "true"),
    ("is baylor and mary hardin baylor the same school", "true"),
    ("can you get the death penalty as a minor", "false"),
    ("did indian football team qualified for fifa 2018", "false"),
    ("are t rex and tyrannosaurus rex the same", "true"),
    ("is the old panama canal still in use", "true"),
    ("do you need a pal to possess ammunition", "true"),
    ("do blue and pink cotton candy taste the same", "false"),
    ("did to kill a mockingbird win an academy award", "true"),
    ("is there such a thing as a floating island", "true"),
    ("do female ferrets die if they don't mate", "true"),
    ("will all xbox 360 games work on xbox one", "false"),
    ("is there a right and left brachiocephalic artery", "false"),
    ("do the runners up on survivor win money", "true"),
    ("is there a sequel to love finds a home", "false"),
    ("will there be a second season of 11.22.63", "false"),
    ("are there nuclear power plants in the us", "true"),
    ("is there a tiebreaker in final set at wimbledon", "false"),
    ("were the twin towers the world trade center", "true"),
    ("did deion sanders ever win a world series", "false"),
    ("is a german shepard the same as an alsatian", "true"),
    ("does a frog jump out of boiling water", "true"),
    ("is it possible to create mass from energy", "true"),
    ("is there a movie with 0 on rotten tomatoes", "true"),
    ("is the jaguar s type rear wheel drive", "true"),
    ("is a tablespoon bigger than a dessert spoon", "true"),
    ("is this the last season of bunk'd", "true"),
    ("does the president live in the white house", "true"),
    ("does the dorsal root ganglion carry sensory input", "true"),
    ("is anne with an e filmed on pei", "true"),
    ("is angular frequency and angular velocity the same", "false"),
    ("can someone die from a bullet shot in the air", "true"),
    ("is salt lake city the biggest city in utah", "true"),
    ("was chasing cars written for grey's anatomy", "false"),
    ("did the girl in the lost world die", "false")
]


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


fewshot = "Answer just \"true\" or \"false\" for the following questions. The answers must be 100% factually accurate.\n\nQ: in the phantom menace is padme the queen\nA: true\n\nQ: does anyone on instagram have 1 billion followers\nA: false\n\nQ: does age have any influence on attentional ability or inattentional blindness\nA: true\n\nQ: you can tell a lot about a person by how they treat their waiter\nA: true\n\nQ: is wheat flour and white flour the same thing\nA: false\n\nQ: does the cat in the hat have a name\nA: false\n\nQ: do you always have to say check in chess\nA: false\n\nQ: is windows movie maker part of windows essentials\nA: true\n\nQ:"
cumul = 0
for pair in qa_pairs:
    response = query({
        "inputs": fewshot + pair[0] + "\nA:"
    })
    if response[0]["generated_text"][(614 +len(pair[0]) + 4):] == pair[1]:
        cumul += 1
    else:
        print("WRONG: " + pair[0])

print("Total accuracy:", cumul/100)
