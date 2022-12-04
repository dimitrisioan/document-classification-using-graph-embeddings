from tkinter import *

root = Tk()
# Windows box dialogs' title
root.title("Experiments")
# Windows box dialogs' icon
# root.iconbitmap("")
# Windows box dialogs' dimensions
root.geometry("500x500")

my_menu = Menu(root)
root.config(menu=my_menu)


# Click "New" command
def new_command():
    pass


# Click "Undo" command
def undo_command():
    pass


# Click "Copy" command
def copy_command():
    pass


# Click "Paste" command
def paste_command():
    pass


# Create a "File" menu item

file_menu = Menu(my_menu)
my_menu.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="New", command=new_command)
file_menu.add_separator()
file_menu.add_command(label="Exit", command=root.quit)

# Create an "Edit" menu item

edit_menu = Menu(my_menu)
my_menu.add_cascade(label="Edit", menu=edit_menu)
edit_menu.add_command(label="Undo", command=undo_command)
edit_menu.add_separator()
edit_menu.add_command(label="Copy", command=copy_command)
edit_menu.add_command(label="Paste", command=paste_command)

root.mainloop()
