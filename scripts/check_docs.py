#!/usr/bin/env python3
import re
import sys
import os
from pathlib import Path
from urllib.parse import unquote

def check_file(file_path):
    print(f"Checking {file_path}...")
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all markdown links: [text](link)
    # This regex is simple and might miss some edge cases but works for most standard markdown
    links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
    
    errors = 0
    base_dir = os.path.dirname(os.path.abspath(file_path))

    for text, link in links:
        # Skip anchors in the same file
        if link.startswith('#'):
            continue
            
        # Skip external links
        if link.startswith(('http://', 'https://', 'mailto:')):
            continue

        # Handle anchors in other files (e.g. file.md#section)
        link_path = link.split('#')[0]
        if not link_path:
            continue

        # Resolve absolute paths (relative to repo root if they start with /)
        # But usually markdown links are relative.
        # Let's assume relative to the file.
        
        target_path = os.path.join(base_dir, link_path)
        # Handle URL decoding (e.g. %20 -> space)
        target_path = unquote(target_path)
        
        if not os.path.exists(target_path):
            print(f"  ❌ Broken link: [{text}]({link}) -> {link_path} not found")
            errors += 1
        else:
            # print(f"  ✅ Link found: {link}")
            pass

    if errors == 0:
        print(f"✅ {file_path} passed checks.")
        return True
    else:
        print(f"❌ {file_path} has {errors} broken links.")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python check_docs.py <file1> <file2> ...")
        sys.exit(1)

    files = sys.argv[1:]
    all_passed = True

    for file in files:
        if not check_file(file):
            all_passed = False
            
    if not all_passed:
        sys.exit(1)

if __name__ == "__main__":
    main()
