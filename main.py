import json
import os

from dotenv import load_dotenv
from groq import Groq


# ============================================================
# 1. SETUP
# ============================================================

load_dotenv()

client = Groq()


# ============================================================
# 2. SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are my Spanish conversation partner and tutor.

Talk to me like a real person texting, NOT like an AI tutor giving a lesson.

STRICT CONVERSATION RULES:
- Usually reply with only 1 sentence.
- Maximum 2 short sentences.
- Ask AT MOST ONE question per response.
- NEVER ask two or more questions.
- NEVER give me a list of things I could talk about.
- NEVER offer multiple lesson options.
- NEVER give tables, headings, bullet points, or long explanations.
- Do not over-explain.
- Keep the conversation moving naturally.
- The learner should talk MORE than you do.

SPANISH LEARNING RULES:
- I am currently a beginner.
- Mix Spanish and English when needed so I can understand you.
- Prefer Spanish words that I already know.
- Do NOT suddenly use lots of unfamiliar Spanish vocabulary.
- Introduce at most ONE important new Spanish word or expression per response.
- When introducing a new Spanish word, briefly give its English meaning.
- Reuse recently learned Spanish words naturally later.
- Gradually use more Spanish as my ability improves.

CORRECTIONS:
- Correct only ONE important mistake at a time.
- Ignore minor accent or spelling mistakes when you can understand me.
- If I use an English word inside a Spanish sentence, naturally tell me
  the Spanish equivalent and continue the conversation.
- Do NOT turn corrections into grammar lessons unless I specifically ask.

EXAMPLES OF THE STYLE I WANT:

User: Hola
Assistant: ¡Hola! ¿Cómo estás?

User: Estoy bien
Assistant: ¡Qué bien! ¿Qué hiciste hoy?

User: I played with Olive
Assistant: "Played" is "jugué". ¿Jugaste con Olive?

User: Si, jugue con Olive
Assistant: ¡Muy bien! ¿Olive suele jugar mucho?

User: what is suele
Assistant: "Suele" means "usually" or "tends to". ¿Olive suele dormir mucho?

User: Olive suele dormir mucho
Assistant: ¡Muy bien! ¿Dónde suele dormir?

IMPORTANT:
This should feel like texting a Spanish-speaking friend who is helping me learn.
Do not behave like a textbook or classroom teacher.
"""


# ============================================================
# 3. CONVERSATION MEMORY FUNCTIONS
# ============================================================

def load_memory():
    """
    Load previous conversation history.

    If no previous conversation exists,
    create a new conversation starting with the system prompt.
    """

    if os.path.exists("conversation.json"):

        with open(
            "conversation.json",
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]


def save_memory(messages):
    """
    Save conversation history to conversation.json.
    """

    with open(
        "conversation.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            messages,
            file,
            ensure_ascii=False,
            indent=4
        )


# ============================================================
# 4. LEARNER PROFILE FUNCTIONS
# ============================================================

def load_learner_profile():
    """
    Load information about the learner's current Spanish ability.
    """

    with open(
        "learner_profile.json",
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def build_learner_context(learner):
    """
    Convert the learner profile into instructions
    that the LLM can understand.
    """

    mastered_words = []
    learning_words = []
    weak_words = []

    for word, mastery in learner["vocabulary"].items():

        if mastery >= 0.8:
            mastered_words.append(word)

        elif mastery >= 0.4:
            learning_words.append(word)

        else:
            weak_words.append(word)

    context = f"""

CURRENT LEARNER STATE:

Current level:
{learner["level"]}

Target proportion of Spanish:
{int(learner["spanish_ratio"] * 100)}%

WORDS MASTERED:
{", ".join(mastered_words)}

WORDS CURRENTLY LEARNING:
{", ".join(learning_words)}

WEAK / NEW WORDS:
{", ".join(weak_words)}

GRAMMAR THE LEARNER STRUGGLES WITH:
{", ".join(learner["weak_grammar"])}


ADAPT YOUR LANGUAGE TO THIS LEARNER:

MASTERED WORDS:
Use these naturally without translating them.

LEARNING WORDS:
Use these regularly so the learner gets repeated exposure.
Only translate them if the learner seems confused.

WEAK WORDS:
If you use one of these words, briefly give its English meaning.

UNKNOWN WORDS:
Avoid unnecessary advanced Spanish vocabulary.
If an unknown Spanish word is necessary, introduce only ONE at a time
and immediately make its meaning understandable.

Do not introduce several unknown Spanish words in the same sentence.

As the learner improves, gradually reduce English and increase Spanish.
"""

    return context


# ============================================================
# 5. LOAD LEARNER + CONVERSATION
# ============================================================

learner = load_learner_profile()

messages = load_memory()

learner_context = build_learner_context(learner)


# Always replace the stored system prompt with the newest version.
#
# This allows us to improve SYSTEM_PROMPT without deleting
# the learner's previous conversation.

messages[0] = {
    "role": "system",
    "content": SYSTEM_PROMPT + learner_context
}


# ============================================================
# 6. START PROGRAM
# ============================================================

print()
print("🇪🇸 Spanish AI")
print("Type 'quit' to stop.")
print()


# ============================================================
# 7. CONVERSATION LOOP
# ============================================================

while True:

    # Get message from learner
    user_input = input("You: ").strip()

    # Ignore empty messages
    if not user_input:
        continue

    # Stop program
    if user_input.lower() == "quit":

        print()
        print("Tutor: ¡Hasta luego! 👋")

        break


    # --------------------------------------------------------
    # Add learner message to conversation
    # --------------------------------------------------------

    messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )


    # --------------------------------------------------------
    # Ask LLM for response
    # --------------------------------------------------------

    try:

        response = client.chat.completions.create(

            model="openai/gpt-oss-20b",

            messages=messages,

            # Keeps responses from becoming unnecessarily long
            max_tokens=150
        )


        tutor_reply = response.choices[0].message.content


    # --------------------------------------------------------
    # Handle API errors without crashing the whole program
    # --------------------------------------------------------

    except Exception as e:

        print()
        print("⚠️ Something went wrong with the LLM request.")
        print("Error:", e)
        print()

        # Remove the user message because this conversation
        # turn failed.
        messages.pop()

        continue


    # --------------------------------------------------------
    # Save tutor response into conversation history
    # --------------------------------------------------------

    messages.append(
        {
            "role": "assistant",
            "content": tutor_reply
        }
    )


    # --------------------------------------------------------
    # Save conversation permanently
    # --------------------------------------------------------

    save_memory(messages)


    # --------------------------------------------------------
    # Display response
    # --------------------------------------------------------

    print()
    print("Tutor:", tutor_reply)
    print()