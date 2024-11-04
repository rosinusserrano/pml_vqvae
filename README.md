# pml_vqvae

## When editing LaTex in VS Code

### Setup

1. Install [TexLive](https://www.tug.org/texlive/) (best would be to install the full version)
2. Install [Latex Workshop](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop) VS Code extension
3. Make sure you have pulled the git version where there is a .vscode/settings.json file containing settings for latex workshop and a gitignore that has the file endings of auxiliary latex build files

### Workflow

#### TL;DR

Open the .tex file you want to edit, press Ctrl + Alt + B (Windows/Linux), Cmd + Option + B (macOS) to build the PDF and then Ctrl + Alt + V (Windows/Linux), Cmd + Option + V (macOS) to view the PDF. Then edit your file and you should get live updates in the PDF view.

#### Shortcuts

Generally, you should be able to open the .tex document that you are interested in and use the following commands to compile, view, and navigate the document:

1. Compile the LaTeX document
    - Shortcut: Ctrl + Alt + B (Windows/Linux), Cmd + Option + B (macOS)
    - Description: Compiles the main .tex file using the default recipe (e.g., latexmk).

1. View PDF Preview
    - Shortcut: Ctrl + Alt + V (Windows/Linux), Cmd + Option + V (macOS)
    - Description: Opens a PDF preview window beside the editor to view the compiled PDF. The preview updates automatically after each successful compilation.

1. Sync PDF with Source (Forward Sync)
    - Shortcut: Ctrl + Alt + J (Windows/Linux), Cmd + Option + J (macOS)
    - Description: Jumps from the .tex source in the editor to the corresponding location in the PDF preview.

1. Sync Source with PDF (Reverse Sync)
    - Shortcut: Ctrl + Click in the PDF Preview
    - Description: Jumps from a location in the PDF preview back to the corresponding place in the .tex source