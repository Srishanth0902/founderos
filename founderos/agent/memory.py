from langgraph.checkpoint.memory import MemorySaver

def get_checkpointer():
    """
    Returns the checkpointer for conversation memory.
    """
    return MemorySaver()

# We can also add helpers to fetch user profile, goals, last discussion here.
def get_user_profile():
    return "User is Srishanth, an ambitious founder focused on practical AI."
