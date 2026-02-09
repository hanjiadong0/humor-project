"""
Quick fix to add .env loading to GPT_joke_generation.ipynb Cell 0
"""

print("Adding .env loading code to Cell 0...")
print("\nInstructions:")
print("1. Open your GPT_joke_generation.ipynb")
print("2. In Cell 0 (imports), ADD these lines after the existing imports:")
print()
print("-" * 80)
print("""
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from project root .env file
load_dotenv(Path('../../.env'))

print(f"✓ Environment loaded")
print(f"✓ API Key: {os.getenv('NVIDIA_API_KEY')[:20] if os.getenv('NVIDIA_API_KEY') else 'NOT FOUND'}...")
""")
print("-" * 80)
print()
print("3. In Cell 2 (generator initialization), CHANGE:")
print("   FROM: api_key='REDACTED_CREDENTIAL'")
print("   TO:   api_key=os.getenv('NVIDIA_API_KEY')")
print()
print("4. Restart kernel and re-run all cells")
print()
print("=" * 80)
print("NOTE: The real problem is that the pipeline returns None")
print("This is because the NVIDIA API doesn't like the short prompts in steps 1-2")
print("The single-step method works fine (as you saw in Example 1)")
print("=" * 80)
