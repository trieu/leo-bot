from datetime import datetime
import random

def self_reflection():
    # Identify personal interests, hobbies, and passions.
    # Reflect on preferred social interactions (small groups, one-on-one, online communities, etc.).
    # Assess time availability for social activities.
    print("From 1 to 10, please rate your loneliness")
    loneliness_level = 5
    try:
        loneliness_level = int(input())
    except:
        print("Please input a valid number from 1 to 10")
    return loneliness_level


def do_activity(activities):
    """ do some activitiws """
    # the minimum number of activities to escape loneliness
    number_activity = 3
    # Get the current time.
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M")
    # Choose an activity from the list and do it.
    activity = random.choices(activities, k=number_activity)
    print(f"\n At [{current_datetime}], I'm going to {activity} to escape loneliness.")
    return self_reflection()


def escape_loneliness():
    """Design an algorithm to escape loneliness."""

    # Create a list of activities that can be done to escape loneliness.
    activities = [
        "Spend time with soulmates, friends and family",
        "Join a club or group that shares your interests",
        "Volunteer your time",
        "Take a class or workshop",
        "Travel",
        "Pet a dog or cat",
        "Read a book",
        "Watch a movie",
        "Listen to music",
        "Meditate",
        "Practice yoga",
    ]

    loneliness_level = do_activity(activities)

    # Check if the loneliness has been reduced.

    if loneliness_level < initial_loneliness_level:
        print("I feel less lonely now.")
    else:
        print("I'm still feeling lonely.")


if __name__ == "__main__":
    # Get the initial loneliness level.
    initial_loneliness_level = 5

    # Escape loneliness.
    escape_loneliness()

    loneliness_level = self_reflection()
    # Check if the loneliness has been reduced.
    if loneliness_level < initial_loneliness_level:
        print("I successfully escaped loneliness!")
    else:
        print("I was not able to escape loneliness.")