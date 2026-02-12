from pathlib import Path
import os

def load_knowledge(knowledge_dir):
    knowledge = []
    for file_path in Path(knowledge_dir).rglob('*.txt'):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            knowledge.append({
                'filename': file_path.name,
                'content': content
            })
    return knowledge

def main():
    knowledge_dir = 'knowledge'
    if not os.path.exists(knowledge_dir):
        print(f"Knowledge directory '{knowledge_dir}' does not exist.")
        return

    knowledge = load_knowledge(knowledge_dir)
    print(f"Loaded {len(knowledge)} documents from the knowledge base.")

if __name__ == "__main__":
    main()