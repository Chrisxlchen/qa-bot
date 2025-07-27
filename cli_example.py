#!/usr/bin/env python3
"""
Simple CLI example for the QA Bot
Usage: python cli_example.py
"""

import os
from dotenv import load_dotenv
from src.qa_bot import QABot

def main():
    load_dotenv()
    
    print("🤖 QA Bot CLI Example")
    print("=" * 40)
    
    # Initialize the bot
    print("Initializing QA Bot...")
    bot = QABot(
        documents_path="./documents",
        persist_directory="./chroma_db",
        embedding_model_type="deepseek",  # or "openai" or "huggingface"
        embedding_model_name="deepseek-embedding",  # or "text-embedding-ada-002" or "all-MiniLM-L6-v2"
        llm_model="deepseek-chat"
    )
    
    # Check if documents need indexing
    stats = bot.get_stats()
    print(f"Current stats: {stats}")
    
    if stats['total_documents'] == 0:
        print("\nNo documents indexed. Indexing documents...")
        bot.index_documents()
    else:
        print(f"\nFound {stats['total_documents']} indexed documents.")
        reindex = input("Do you want to re-index? (y/N): ").lower().strip()
        if reindex == 'y':
            bot.index_documents(force_reindex=True)
    
    # Interactive Q&A loop
    print("\n" + "=" * 40)
    print("You can now ask questions! Type 'quit' to exit.")
    print("=" * 40)
    
    while True:
        question = input("\n❓ Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if not question:
            continue
        
        print("🔍 Searching for answers...")
        result = bot.ask(question, n_results=5)
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            continue
        
        print(f"\n🎯 Answer:")
        print(result['answer'])
        
        if result.get('sources'):
            print(f"\n📚 Sources:")
            for i, source in enumerate(result['sources'], 1):
                print(f"  {i}. {source}")
        
        print(f"\n📊 Used {result.get('context_used', 0)} document chunks")

if __name__ == "__main__":
    main()