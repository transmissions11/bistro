import sys
import requests

print(f"Loading {sys.argv[1]}.py from GitHub...")

script = requests.get(
    f"https://raw.githubusercontent.com/Lightning-AI/lit-gpt/main/scripts/{sys.argv[1]}.py"
).text

sys.argv = sys.argv[1:] + sys.argv[2:]  # Remove the first argument.

exec(
    script,
    {"__file__": __file__, "__name__": __name__},
)
