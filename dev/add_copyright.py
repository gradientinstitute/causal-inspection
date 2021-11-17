"""
Script to insert Gradient institute cpwrite statement into all python files.
"""

import os
from typing import List

COPYRIGHT = "Copyright (C) 2019-2021 Gradient Institute Ltd."
TAG = '"""'


def split_docstring_from_content(content: List[str]):
    """Extract the Docstring (and remove any existing Gradient Copyright statement)"""
    module_doc = []
    reading_module_doc = False
    for i, line in enumerate(content):
        line = line.strip()
        if line.startswith(TAG):
            if line.endswith(TAG) and len(line) > 3:
                return [line.strip(TAG)], content[i+1:]
            elif reading_module_doc:
                return module_doc, content[i+1:]
            else:
                reading_module_doc = True
                module_doc.append(line.strip(TAG))
        elif reading_module_doc:
            module_doc.append(line.strip(TAG))
            if line.endswith(TAG):
                return module_doc, content[i+1:]

    return [], content


def _update_doc(doc):
    updated = []
    for line in doc:
        if len(line) > 0 and "copyright" not in line.lower():
            updated.append(line)
        if "copyright" in line.lower() and "gradient" not in line.lower():
            raise ValueError("Non-gradient copyright line found")
    updated.append(COPYRIGHT)
    updated = "\n".join(updated)
    return f'{TAG}\n{updated}\n{TAG}\n'


def add_copyright():

    for root, dirs, files in os.walk("..", topdown=True):
        files = [f for f in files if not f[0] == '.']
        dirs[:] = [d for d in dirs if not d[0] == '.']
        for f in files:
            if f.endswith(".py") and not f.startswith("__init__."):
                filepath = os.path.join(root, f)
                with open(filepath, 'r') as file_in:
                    print(f)
                    content = file_in.readlines()
                    doc, remainder = split_docstring_from_content(content)
                    doc = _update_doc(doc)
                with open(filepath, 'w') as file_out:
                    file_out.write(doc)
                    file_out.write("\n")
                    file_out.write(''.join(remainder))
                    # print(_update_doc(doc))


if __name__ == "__main__":
    add_copyright()
